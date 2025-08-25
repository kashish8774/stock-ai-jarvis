import type { NextApiRequest, NextApiResponse } from "next";
import yahooFinance from "yahoo-finance2";

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const symbol = req.query.symbol?.toString() || "AAPL";

  try {
    const quote = await yahooFinance.quote(symbol);
    const hist = await yahooFinance.historical(symbol, { period1: "2023-01-01" });

    res.status(200).json({
      symbol: symbol,
      price: quote.regularMarketPrice,
      history: hist.map((h) => ({
        date: h.date.toISOString().split("T")[0],
        close: h.close,
      })),
    });
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch data", details: err });
  }
}
