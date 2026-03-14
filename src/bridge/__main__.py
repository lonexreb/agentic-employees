"""Allow running bridge as: python -m src.bridge"""

from src.bridge.service import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
