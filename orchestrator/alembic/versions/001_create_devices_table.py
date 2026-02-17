"""Create devices table

Revision ID: 001
Revises:
Create Date: 2026-02-17

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("device_model", sa.String(255), nullable=False),
        sa.Column("os_version", sa.String(50), nullable=False),
        sa.Column("chip", sa.String(100)),
        sa.Column("memory_bytes", sa.BigInteger),
        sa.Column("cpu_cores", sa.Integer),
        sa.Column("gpu_cores", sa.Integer),
        sa.Column("neural_engine_cores", sa.Integer),
        sa.Column("battery_level", sa.Float),
        sa.Column("battery_state", sa.String(20)),
        sa.Column("status", sa.String(20), nullable=False, server_default="online"),
        sa.Column("metrics", JSON),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("notes", sa.Text),
    )


def downgrade() -> None:
    op.drop_table("devices")
