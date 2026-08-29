import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 800})
        await page.goto('http://localhost:8501', wait_until='networkidle')
        await asyncio.sleep(5)
        await page.click('text=Audit Ledger')
        await asyncio.sleep(2)
        await page.screenshot(path='docs/screenshots/audit_ledger.png')
        print('Took audit ledger screenshot')
        await page.click('text=System & Policy Analytics')
        await asyncio.sleep(2)
        await page.screenshot(path='docs/screenshots/analytics.png')
        print('Took analytics screenshot')
        await browser.close()

asyncio.run(main())
