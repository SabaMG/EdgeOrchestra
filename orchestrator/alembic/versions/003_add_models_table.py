"""Add models table and model_id FK on training_jobs

Revision ID: 003
Revises: 002
Create Date: 2026-02-19

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("architecture", sa.String(50), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="initial"),
        sa.Column("parent_model_id", UUID(as_uuid=True), sa.ForeignKey("models.id"), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.add_column(
        "training_jobs",
        sa.Column("model_id", UUID(as_uuid=True), sa.ForeignKey("models.id"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_jobs", "model_id")
    op.drop_table("models")
