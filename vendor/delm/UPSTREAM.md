# Upstream DeLM Source

- **Repository**: `https://github.com/yuzhenmao/DeLM`
- **Commit**: `66617616e708758f91624e5cc03659732b82f6a6`
- **License**: MIT License (see `vendor/delm/LICENSE`)

All files under `vendor/delm/board/` are unmodified copies of upstream `board/`:

- `vendor/delm/board/__init__.py`
- `vendor/delm/board/shared_context.py`
- `vendor/delm/board/protected_board.py`
- `vendor/delm/board/context_client.py`
- `vendor/delm/board/context.sh`

Verified clean against upstream `board/` (`diff -ru --exclude=__pycache__` returns exit code `0` with empty diff).
