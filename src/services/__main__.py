"""Allow running training service as: python -m src.services.training"""

from src.services.training import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
