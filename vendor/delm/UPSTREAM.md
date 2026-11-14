# Upstream DeLM Source

- **Repository**: `https://github.com/yuzhenmao/DeLM`
- **Commit**: `66617616e708758f91624e5cc03659732b82f6a6`
- **License**: MIT License (see `vendor/delm/LICENSE`)

All files under `vendor/delm/board/` and `vendor/delm/prompts/` are unmodified copies of upstream `board/` and `prompts/`:

- `vendor/delm/board/__init__.py`
- `vendor/delm/board/shared_context.py`
- `vendor/delm/board/protected_board.py`
- `vendor/delm/board/context_client.py`
- `vendor/delm/board/context.sh`
- `vendor/delm/prompts/__init__.py`
- `vendor/delm/prompts/prompts_n2.py`

Verified clean against upstream `board/` and `prompts/{__init__.py,prompts_n2.py}` (`diff -u` returns exit code `0` with empty diff).
