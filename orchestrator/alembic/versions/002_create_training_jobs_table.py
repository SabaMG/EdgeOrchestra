"""Create training_jobs table

Revision ID: 002
Revises: 001
Create Date: 2026-02-17

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON, UUID

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "training_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("num_rounds", sa.Integer, nullable=False, server_default="5"),
        sa.Column("current_round", sa.Integer, nullable=False, server_default="0"),
        sa.Column("min_devices", sa.Integer, nullable=False, server_default="1"),
        sa.Column("learning_rate", sa.Float, nullable=False, server_default="0.01"),
        sa.Column("round_metrics", JSON),
        sa.Column("config", JSON),
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
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("training_jobs")
