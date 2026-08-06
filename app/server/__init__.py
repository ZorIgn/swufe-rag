"""Single public HTTP application for the canonical academic agent."""

from app.server.canonical import AcademicAuditRequest, AskRequest, ErrorResponse, create_app

app = create_app()


__all__ = ["AcademicAuditRequest", "AskRequest", "ErrorResponse", "app", "create_app"]
