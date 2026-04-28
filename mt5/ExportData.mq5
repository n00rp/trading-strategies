//+------------------------------------------------------------------+
//| ExportData.mq5 - Export OHLCV data for multiple symbols/timeframes
//+------------------------------------------------------------------+
#property copyright "Trading Strategies"
#property version   "1.00"
#property script_show_inputs

input int MaxBars = 100000; // Max bars to fetch per timeframe

//+------------------------------------------------------------------+
void OnStart()
{
   string symbols[] = {"US500", "USTEC", "US30", "DE40"};
   string names[]   = {"sp500", "nasdaq", "dowjones", "dax"};
   ENUM_TIMEFRAMES tfs[] = {PERIOD_M1, PERIOD_M5, PERIOD_M10, PERIOD_M15, PERIOD_M30, PERIOD_H1, PERIOD_D1};
   string tfNames[] = {"1m", "5m", "10m", "15m", "30m", "1h", "1d"};
   
   string dataPath = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";
   
   for(int s = 0; s < ArraySize(symbols); s++)
   {
      // Check if symbol exists, try alternatives
      string sym = symbols[s];
      if(!SymbolSelect(sym, true))
      {
         // Try .cash suffix
         if(SymbolSelect(sym + ".cash", true))
            sym = sym + ".cash";
         else
         {
            Print("Symbol not found: ", symbols[s]);
            continue;
         }
      }
      
      Print("Processing: ", sym, " (", names[s], ")");
      
      for(int t = 0; t < ArraySize(tfs); t++)
      {
         MqlRates rates[];
         int copied = CopyRates(sym, tfs[t], 0, MaxBars, rates);
         
         if(copied <= 0)
         {
            Print("  ", tfNames[t], ": no data (error ", GetLastError(), ")");
            continue;
         }
         
         string filename = names[s] + "_" + tfNames[t] + ".csv";
         int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
         
         if(handle == INVALID_HANDLE)
         {
            Print("  Failed to create ", filename);
            continue;
         }
         
         FileWrite(handle, "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume");
         
         for(int i = 0; i < copied; i++)
         {
            FileWrite(handle,
               (long)rates[i].time,
               DoubleToString(rates[i].open, 5),
               DoubleToString(rates[i].high, 5),
               DoubleToString(rates[i].low, 5),
               DoubleToString(rates[i].close, 5),
               (long)rates[i].tick_volume,
               rates[i].spread,
               (long)rates[i].real_volume);
         }
         
         FileClose(handle);
         Print("  ", tfNames[t], ": ", copied, " bars -> ", filename);
      }
   }
   
   Print("Export complete!");
}
