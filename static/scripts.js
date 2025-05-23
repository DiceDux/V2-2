async function fetchMode() {
    const res = await fetch("/api/mode");
    const data = await res.json();
    document.getElementById("mode").textContent = data.mode.toUpperCase();
  }
  
  async function fetchBalances() {
    const res = await fetch("/api/balances");
    const data = await res.json();
    const tbody = document.getElementById("balances-body");
    tbody.innerHTML = "";
  
    data.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.symbol}</td>
        <td>${parseFloat(row.balance).toFixed(4)}</td>
        <td>${row.updated_at}</td>
      `;
      tbody.appendChild(tr);
    });
  }
  
  async function fetchTrades() {
    const res = await fetch("/api/trades");
    const data = await res.json();
    const tbody = document.getElementById("trades-body");
    tbody.innerHTML = "";
  
    data.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.id}</td>
        <td>${row.symbol}</td>
        <td>${row.action.toUpperCase()}</td>
        <td>${parseFloat(row.entry_price).toFixed(4)}</td>
        <td>${row.confidence}</td>
        <td>${row.profit !== null ? parseFloat(row.profit).toFixed(2) + "%" : "-"}</td>
        <td>${row.timestamp}</td>
        <td>${row.mode.toUpperCase()}</td>
      `;
      if (row.action.toLowerCase() === "buy") {
        tr.classList.add(row.action.toLowerCase() + "-row"); 
      } else if (row.action.toLowerCase() === "sell") {
        tr.classList.add(row.action.toLowerCase() + "-row");
      }      
      tbody.appendChild(tr);      
    });    
  }
  
  // بارگذاری اولیه
  fetchMode();
  fetchBalances();
  fetchTrades();
  
  // بروزرسانی هر 10 ثانیه
  setInterval(() => {
    fetchMode();
    fetchBalances();
    fetchTrades();
    fetchOpenTrades(); // ✅ این خط جدید اضافه بشه
  }, 10000);
  
  // لاگ زنده
  function loadLogs() {
    fetch("/api/logs")
      .then(response => response.text())
      .then(data => {
        const logBox = document.getElementById("live-log");
        logBox.innerHTML = data.replace(/\n/g, "<br>");
        logBox.scrollTop = logBox.scrollHeight;
      })
      .catch(err => {
        document.getElementById("live-log").textContent = "❌ خطا در دریافت لاگ";
      });
  }
  
  setInterval(loadLogs, 5000);
  loadLogs();

  async function closeAllTrades() {
    const resultBox = document.getElementById("close-result");
    resultBox.textContent = "⏳ در حال بستن معاملات...";
    try {
      const res = await fetch("/api/close_all_trades", { method: "POST" });
      const data = await res.json();
      resultBox.textContent = data.message;
  
      // بروزرسانی سریع بعد از بستن
      fetchTrades();
      fetchBalances();
    } catch (err) {
      resultBox.textContent = "❌ خطا در بستن معاملات!";
    }
  }
  
  function updateTradeStatus(newStatus) {
    fetch(`/api/trade-status/${newStatus}`, { method: "POST" })
      .then(res => res.json())
      .then(data => {
        document.getElementById("status-msg").textContent = data.message;
        fetchMode(); // بروزرسانی وضعیت
      })
      .catch(err => {
        document.getElementById("status-msg").textContent = "❌ خطا در تغییر وضعیت.";
      });
  }
  
// تابع جدید برای گرفتن نتایج بک‌تست
async function fetchBacktestResults() {
  const res = await fetch("/api/backtest-results");
  const data = await res.json();
  const tbody = document.getElementById("backtest-results-body");
  tbody.innerHTML = "";

  data.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.symbol}</td>
      <td>${row.total_trades}</td>
      <td>${(row.win_rate * 100).toFixed(2)}%</td>
      <td>${row.total_profit.toFixed(2)}</td>
      <td>${row.final_balance.toFixed(2)}</td>
      <td>${row.executed_at}</td>
    `;
    tbody.appendChild(tr);
  });
}

// اصلاح تابع loadChart
async function loadChart() {
  const symbol = document.getElementById("symbol-select").value;

  const [candlesRes, tradesRes] = await Promise.all([
    fetch(`/api/candles?symbol=${symbol}`),
    fetch(`/api/trades?symbol=${symbol}`)
  ]);

  const candleData = await candlesRes.json();
  const tradeData = await tradesRes.json();

  const dates = candleData.map(c => new Date(c.timestamp * 1000));
  const open = candleData.map(c => c.open);
  const high = candleData.map(c => c.high);
  const low = candleData.map(c => c.low);
  const close = candleData.map(c => c.close);

  const candles = {
    x: dates,
    open: open,
    high: high,
    low: low,
    close: close,
    type: "candlestick",
    name: symbol
  };

  const buys = tradeData.filter(t => t.action.toLowerCase() === "buy");
  const sells = tradeData.filter(t => t.action.toLowerCase() === "sell");

  const buySignals = {
    x: buys.map(t => new Date(t.timestamp)),
    y: buys.map(t => t.entry_price),
    mode: "markers+text",
    type: "scatter",
    name: "Buy",
    marker: { color: "green", size: 10, symbol: "triangle-up" },
    textposition: "bottom center",
    text: buys.map(t => "Buy")
  };

  const sellSignals = {
    x: sells.map(t => new Date(t.timestamp)),
    y: sells.map(t => t.entry_price),
    mode: "markers+text",
    type: "scatter",
    name: "Sell",
    marker: { color: "red", size: 10, symbol: "triangle-down" },
    textposition: "top center",
    text: sells.map(t => {
      if (t.profit !== null && t.profit !== undefined) {
        const p = parseFloat(t.profit).toFixed(2);
        return `${p > 0 ? "+" : ""}${p}%`;
      }
      return "Sell";
    })
  };

  const layout = {
    title: `نمودار قیمت ${symbol}`,
    xaxis: { title: "زمان" },
    yaxis: { title: "قیمت" }
  };

  Plotly.newPlot("chart", [candles, buySignals, sellSignals], layout);
}

// بارگذاری اولیه
fetchMode();
fetchBalances();
fetchTrades();
fetchOpenTrades();
fetchBacktestResults(); // اضافه کردن بک‌تست

// بروزرسانی هر 10 ثانیه
setInterval(() => {
  fetchMode();
  fetchBalances();
  fetchTrades();
  fetchOpenTrades();
  fetchBacktestResults(); // اضافه کردن بک‌تست
}, 10000);

  
  async function fetchOpenTrades() {
    const res = await fetch("/api/open_trades");
    const data = await res.json();
    const tbody = document.getElementById("open-trades-body");
    tbody.innerHTML = "";
  
    data.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.symbol}</td>
        <td>${row.action.toUpperCase()}</td>
        <td>${parseFloat(row.entry_price).toFixed(8)}</td>
        <td>${row.timestamp}</td>
        <td>${parseFloat(row.tp_price).toFixed(8)}</td>
        <td>${parseFloat(row.sl_price).toFixed(8)}</td>
        <td>${row.tp_step}</td>
        <td>${parseFloat(row.last_price).toFixed(8)}</td>
        <td style="color: ${row.live_profit >= 0 ? 'limegreen' : 'red'}">
          ${row.live_profit !== null && row.live_profit !== undefined ? row.live_profit.toFixed(2) + "%" : "-"}
          </td>        
      `;    
      tbody.appendChild(tr);
    });
  }
  
  async function loadBalanceChart() {
    const res = await fetch("/api/performance-chart");
    const data = await res.json();
  
    const x = data.map(item => new Date(item.updated_at));
    const y = data.map(item => item.balance);
  
    const trace = {
      x: x,
      y: y,
      type: "scatter",
      mode: "lines+markers",
      name: "Balance",
      line: { color: "blue" }
    };
  
    const layout = {
      title: "روند تغییر موجودی حساب",
      xaxis: { title: "زمان" },
      yaxis: { title: "مقدار موجودی ($)" }
    };
  
    Plotly.newPlot("balance-chart", [trace], layout);
  }  

  // بارگذاری پیش‌فرض
  setTimeout(loadChart, 1000);
  setTimeout(loadBalanceChart, 2000);

  