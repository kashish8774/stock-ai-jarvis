"use client";
import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";

export default function Home() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetch("/api/stock?symbol=AAPL")
      .then((res) => res.json())
      .then((d) => setData(d));
  }, []);

  return (
    <main className="p-6">
      <h1 className="text-2xl font-bold mb-4">📈 AI Stock Jarvis</h1>
      {data ? (
        <div className="bg-white shadow-lg p-4 rounded-xl">
          <h2 className="text-xl font-semibold">{data.symbol} - {data.price}</h2>
          <Line
            data={{
              labels: data.history.map((h: any) => h.date),
              datasets: [
                {
                  label: "Price",
                  data: data.history.map((h: any) => h.close),
                  borderColor: "blue",
                },
              ],
            }}
          />
        </div>
      ) : (
        <p>Loading...</p>
      )}
    </main>
  );
}
