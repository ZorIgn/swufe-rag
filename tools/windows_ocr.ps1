[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PdfPath,

    [string]$OutputPath,

    [ValidateRange(120, 300)]
    [int]$Dpi = 220
)

$ErrorActionPreference = "Stop"

function Await-WinRt {
    param(
        [Parameter(Mandatory = $true)]$Operation,
        [Parameter(Mandatory = $true)][Type]$ResultType
    )
    $method = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq "AsTask" -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1
    })[0]
    $task = $method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    return $task.GetAwaiter().GetResult()
}

$pdf = (Resolve-Path -LiteralPath $PdfPath).Path
if ([IO.Path]::GetExtension($pdf).ToLowerInvariant() -ne ".pdf") {
    throw "PdfPath must point to a PDF file."
}

if (-not $OutputPath) {
    $OutputPath = "$pdf.ocr.json"
}
$output = [IO.Path]::GetFullPath($OutputPath)
$outputDirectory = [IO.Path]::GetDirectoryName($output)
[IO.Directory]::CreateDirectory($outputDirectory) | Out-Null

$pdftoppm = Get-Command pdftoppm.exe -ErrorAction SilentlyContinue
if ($null -eq $pdftoppm) {
    throw "pdftoppm.exe is required to render PDF pages before Windows OCR."
}

$tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$tempRoot = [IO.Path]::GetFullPath((Join-Path $tempBase ("swufe-rag-ocr-" + [Guid]::NewGuid().ToString("N"))))
if (-not $tempRoot.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to create an OCR workspace outside the system temp directory."
}
[IO.Directory]::CreateDirectory($tempRoot) | Out-Null

try {
    $prefix = Join-Path $tempRoot "page"
    & $pdftoppm.Source -q -png -r $Dpi $pdf $prefix
    if ($LASTEXITCODE -ne 0) {
        throw "pdftoppm failed with exit code $LASTEXITCODE."
    }
    $images = Get-ChildItem -File -LiteralPath $tempRoot -Filter "page-*.png" | Sort-Object {
        if ($_.BaseName -match "(\d+)$") { [int]$Matches[1] } else { [int]::MaxValue }
    }
    if (-not $images) {
        throw "No page images were rendered from the PDF."
    }

    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
    [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
    [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
    [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
    [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime] | Out-Null

    $language = [Windows.Globalization.Language]::new("zh-Hans")
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
    if ($null -eq $engine) {
        throw "The Windows zh-Hans OCR language pack is unavailable."
    }

    $pages = @()
    $pageNumber = 0
    foreach ($image in $images) {
        $pageNumber += 1
        $file = Await-WinRt ([Windows.Storage.StorageFile]::GetFileFromPathAsync($image.FullName)) ([Windows.Storage.StorageFile])
        $stream = Await-WinRt ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        try {
            $decoder = Await-WinRt ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
            $bitmap = Await-WinRt ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
            try {
                $result = Await-WinRt ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
                $text = (($result.Lines | ForEach-Object { $_.Text.Trim() }) -join "`n").Trim()
                if (-not $text) {
                    throw "OCR produced no text for page $pageNumber."
                }
                $pages += [ordered]@{ page = $pageNumber; text = $text }
            }
            finally {
                if ($null -ne $bitmap) { $bitmap.Dispose() }
            }
        }
        finally {
            $stream.Dispose()
        }
    }

    $payload = [ordered]@{
        schema_version = 1
        source_pdf = [IO.Path]::GetFileName($pdf)
        engine = "Windows.Media.Ocr/zh-Hans"
        dpi = $Dpi
        pages = $pages
    }
    $json = $payload | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText($output, $json + "`n", [Text.UTF8Encoding]::new($false))
    Write-Output $output
}
finally {
    $resolvedTemp = [IO.Path]::GetFullPath($tempRoot)
    if ($resolvedTemp.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $resolvedTemp)) {
        Remove-Item -Recurse -Force -LiteralPath $resolvedTemp
    }
}
