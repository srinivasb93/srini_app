"""
TradingView Lightweight Charts integration with NiceGUI

This module renders a chart page using TradingView's Lightweight Charts
library directly via JS for high-performance, responsive charts.

Usage:
    await render_tradingview_page(fetch_api, user_storage, instruments)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
import uuid
from typing import Dict, List, Any

import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)


def build_lw_js_safe(chart_id: str, overlay_volume: bool, candle_data: list, volume_data: list) -> str:
    """Build JS with placeholders to avoid f-string brace issues."""
    candle_json = json.dumps(candle_data)
    volume_json = json.dumps(volume_data)
    overlay_str = 'true' if overlay_volume else 'false'
    template = r"""
        (function() {
          window._lwCharts = window._lwCharts || {};
          const el = document.getElementById('__CHART_ID__');
          if (!el) return;

          function removeExistingLib() {
            const scripts = Array.from(document.getElementsByTagName('script'));
            scripts.forEach(s => {
              const src = s.getAttribute('src') || '';
              if (src.includes('lightweight-charts')) {
                try { s.parentNode && s.parentNode.removeChild(s); } catch(e) {}
              }
            });
            try { delete window.LightweightCharts; } catch(e) { window.LightweightCharts = undefined; }
          }

          function start(attempt) {
            attempt = attempt || 0;
            if (typeof window.LightweightCharts === 'undefined' || typeof window.LightweightCharts.createChart !== 'function') {
              if (attempt === 0) {
                removeExistingLib();
                const s = document.createElement('script');
                // Prefer official CDN first; fallback to local static if blocked
                s.src = 'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js';
                s.onerror = function() {
                  const c = document.createElement('script');
                  c.src = '/static/lightweight-charts.standalone.production.js';
                  document.head.appendChild(c);
                };
                document.head.appendChild(s);
              }
              if (attempt < 40) {
                return setTimeout(function() { start(attempt + 1); }, 150);
              } else {
                el.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:#94a3b8;">Lightweight Charts library not loaded</div>';
                return;
              }
            }

            // Probe instance API (detect wrong build) and attempt to repair by reloading from CDN
            try {
              const tmp = document.createElement('div');
              const probe = window.LightweightCharts.createChart(tmp);
              const ok = probe && (typeof probe.addLineSeries === 'function' || typeof probe.addCandlestickSeries === 'function');
              if (!ok) {
                if (attempt < 40) {
                  removeExistingLib();
                  const s = document.createElement('script');
                  s.src = 'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js';
                  document.head.appendChild(s);
                  return setTimeout(function() { start(attempt + 1); }, 150);
                } else {
                  el.innerHTML = '<div style="padding:1rem;color:#ef4444;background:#1f2937;border-radius:6px;">Chart API methods missing (addCandlestickSeries/addLineSeries). Please ensure the official standalone build is loaded.</div>';
                  return;
                }
              }
            } catch (e) {
              if (attempt < 40) {
                removeExistingLib();
                const s = document.createElement('script');
                s.src = 'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js';
                document.head.appendChild(s);
                return setTimeout(function() { start(attempt + 1); }, 150);
              } else {
                el.innerHTML = '<div style="padding:1rem;color:#ef4444;background:#1f2937;border-radius:6px;">Failed to initialize chart. Library may be corrupted.</div>';
                return;
              }
            }

            let ctx = window._lwCharts['__CHART_ID__'];
            if (!ctx) {
              const chart = LightweightCharts.createChart(el, {
                rightPriceScale: { visible: true },
                layout: { background: { type: 'solid', color: '#0a0a0a' }, textColor: '#ddd' },
                grid: { vertLines: { color: 'rgba(100, 100, 100, 0.1)' }, horzLines: { color: 'rgba(100, 100, 100, 0.1)' } },
                crosshair: { mode: 1 },
                timeScale: { rightOffset: 6, borderVisible: true },
                autoSize: true
              });
              // Capability detection and safe fallbacks
              const supportsCandle = typeof chart.addCandlestickSeries === 'function';
              const supportsLine = typeof chart.addLineSeries === 'function';
              const supportsHist = typeof chart.addHistogramSeries === 'function';

              if (!supportsCandle && !supportsLine) {
                el.innerHTML = '<div style="padding:1rem;color:#ef4444;background:#1f2937;border-radius:6px;">Chart API methods missing (addCandlestickSeries/addLineSeries). Please replace the JS with the official TradingView Lightweight Charts standalone build.</div>';
                return;
              }

              const candle = supportsCandle
                ? chart.addCandlestickSeries({ priceLineVisible: true })
                : chart.addLineSeries({ priceLineVisible: true });

              const volume = supportsHist
                ? chart.addHistogramSeries({
                    priceScaleId: __OVERLAY__ ? '' : 'left',
                    priceFormat: { type: 'volume' },
                    priceLineVisible: false,
                    scaleMargins: __OVERLAY__ ? { top: 0.8, bottom: 0 } : { top: 0.7, bottom: 0 }
                  })
                : (supportsLine ? chart.addLineSeries({ priceScaleId: __OVERLAY__ ? '' : 'left' }) : null);
              ctx = window._lwCharts['__CHART_ID__'] = { chart, candle, volume };
              new ResizeObserver(() => chart.resize(el.clientWidth, el.clientHeight)).observe(el);
            }

            // Prepare fallback series data if needed
            const cIn = __CANDLE_DATA__;
            const closeLine = cIn.map(d => ({ time: d.time, value: d.close }));
            const vIn = __VOLUME_DATA__;
            const vLine = vIn.map(d => ({ time: d.time, value: d.value }));

            const candleSetter = (ctx.candle.setData ? ctx.candle.setData.bind(ctx.candle) : null);
            const volumeSetter = (ctx.volume.setData ? ctx.volume.setData.bind(ctx.volume) : null);
            if (candleSetter) {
              try { candleSetter(cIn); } catch(e) { candleSetter(closeLine); }
            }
            if (volumeSetter) {
              try { volumeSetter(vIn); } catch(e) { volumeSetter(vLine); }
            }
            if (__OVERLAY__) {
              ctx.volume.applyOptions({ priceScaleId: '' });
            } else {
              ctx.volume.applyOptions({ priceScaleId: 'left' });
            }
            ctx.chart.timeScale().fitContent();
          }
          start(0);
        })();
    """
    return (
        template
        .replace('__CHART_ID__', chart_id)
        .replace('__OVERLAY__', overlay_str)
        .replace('__CANDLE_DATA__', candle_json)
        .replace('__VOLUME_DATA__', volume_json)
    )
def _build_lw_js(chart_id: str, overlay_volume: bool, candle_data: list, volume_data: list) -> str:
    """Build robust JS to load Lightweight Charts with polling and fallback.
    Avoids immediate failure if the library isn't parsed yet."""
    return """
        (function() {{
          window._lwCharts = window._lwCharts || {{}};
          const el = document.getElementById('{chart_id}');
          if (!el) return;

          function start(attempt) {{
            attempt = attempt || 0;
            if (typeof window.LightweightCharts === 'undefined') {{
              if (attempt === 0) {{
                const s = document.createElement('script');
                s.src = '/static/lightweight-charts.standalone.production.js';
                s.onerror = function() {{
                  const c = document.createElement('script');
                  c.src = 'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js';
                  document.head.appendChild(c);
                }};
                document.head.appendChild(s);
              }}
              if (attempt < 40) {{
                return setTimeout(function() {{ start(attempt + 1); }}, 150);
              }} else {{
                el.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:#94a3b8;">Lightweight Charts library not loaded</div>';
                return;
              }}
            }}

            // Initialize chart once per container
            let ctx = window._lwCharts['{chart_id}'];
            if (!ctx) {{
              const chart = LightweightCharts.createChart(el, {{
                rightPriceScale: {{ visible: true }},
                layout: {{ background: {{ type: 'solid', color: '#0a0a0a' }}, textColor: '#ddd' }},
                grid: {{ vertLines: {{ color: 'rgba(100, 100, 100, 0.1)' }}, horzLines: {{ color: 'rgba(100, 100, 100, 0.1)' }} }},
                crosshair: {{ mode: 1 }},
                timeScale: {{ rightOffset: 6, borderVisible: true }},
                autoSize: true,
              }});
              const candle = chart.addCandlestickSeries({{
                priceLineVisible: true,
              }});
              const volume = chart.addHistogramSeries({{
                priceScaleId: {"''" if overlay_volume else "'left'"},
                priceFormat: {{ type: 'volume' }},
                priceLineVisible: false,
                scaleMargins: { json.dumps({'top': 0.8, 'bottom': 0}) if overlay_volume else json.dumps({'top': 0.7, 'bottom': 0}) }
              }});
              ctx = window._lwCharts['{chart_id}'] = {{ chart, candle, volume }};

              // Responsive resize
              new ResizeObserver(() => chart.resize(el.clientWidth, el.clientHeight)).observe(el);
            }}

            // Update series data
            ctx.candle.setData({json.dumps(candle_data)});
            ctx.volume.setData({json.dumps(volume_data)});

            // Ensure scale options for overlay toggle
            if ({'true' if overlay_volume else 'false'}) {{
              ctx.volume.applyOptions({{ priceScaleId: '' }});
            }} else {{
              ctx.volume.applyOptions({{ priceScaleId: 'left' }});
            }}

            ctx.chart.timeScale().fitContent();
          }}
          start(0);
        })();
    """

def _to_iso_time(ts) -> str:
    try:
        if isinstance(ts, (int, float)):
            # seconds -> ISO string
            return datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%dT%H:%M:%S')
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S')
        # already string-like
        return str(ts)
    except Exception:
        return str(ts)


def _compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha['open'] = (df['open'].shift(1).fillna(df['open']) + df['close'].shift(1).fillna(df['close'])) / 2.0
    ha['high'] = df[['high', 'open', 'close']].max(axis=1)
    ha['low'] = df[['low', 'open', 'close']].min(axis=1)
    return ha[['open', 'high', 'low', 'close']]


async def render_tradingview_page(fetch_api, user_storage, instruments: Dict[str, str]):
    """Render a NiceGUI page that uses TradingView Lightweight Charts.

    Args:
        fetch_api: async callable to fetch backend endpoints
        user_storage: NiceGUI storage for user (theme, broker, etc.)
        instruments: mapping of symbol -> instrument token
    """

    # Do not pre-inject scripts; loader JS will inject CDN then local fallback

    # Shell layout: sidebar (left) + chart area (right)
    with ui.row().classes('w-full gap-2 items-start no-wrap').style('padding: 8px; display:flex; flex-wrap: nowrap; align-items: flex-start;'):
        # Sidebar
        with ui.column().classes('q-pa-sm').style('width: 320px; min-width: 300px; max-width: 360px; background: rgba(15,23,42,0.7); border: 1px solid #334155; border-radius: 10px;'):
            ui.label('Stock Analysis').classes('text-subtitle1').style('color: #e2e8f0;')
            # Data selectors
            if not instruments:
                ui.label('No instruments available').classes('text-red-500 text-caption')
                symbol_select = ui.select(options=[], with_input=True, value=None, label='Symbol').classes('w-full').props('disabled')
            else:
                symbol_select = ui.select(options=sorted(list(instruments.keys())), with_input=True,
                                          value=list(instruments.keys())[0], label='Symbol').classes('w-full').style(
                                              'background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;')
                symbol_select.on('update:model-value', lambda: ui.timer(0.05, lambda: update_chart(), once=True))
            timeframe = ui.select(options=['minute', 'day', 'week', 'month'], value='day', label='Timeframe').classes('w-full').style('background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;')
            timeframe.on('update:model-value', lambda: ui.timer(0.05, lambda: update_chart(), once=True))
            from_date = ui.input('From', value='2023-01-01').props('dense type=date').classes('w-full').style('background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;')
            to_date = ui.input('To', value=datetime.now().strftime('%Y-%m-%d')).props('dense type=date').classes('w-full').style('background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;')
            from_date.on('update:model-value', lambda: ui.timer(0.05, lambda: update_chart(), once=True))
            to_date.on('update:model-value', lambda: ui.timer(0.05, lambda: update_chart(), once=True))
            # Replay controls (stubs)
            with ui.expansion('Replay Data').classes('w-full'):
                replay_switch = ui.switch('Enable Replay', value=False)
                replay_date = ui.input('Replay Date', value=datetime.now().strftime('%Y-%m-%d')).props('type=date').classes('w-full')
                replay_speed = ui.slider(min=0.1, max=2.0, value=0.5, step=0.1).props('label-always').classes('w-full')
                with ui.row().classes('gap-2'):
                    ui.button('Previous Day', icon='chevron_left').on('click', lambda: ui.notify('Prev day (stub)'))
                    ui.button('Next Day', icon='chevron_right').on('click', lambda: ui.notify('Next day (stub)'))
                    ui.button('Play/Pause', icon='play_arrow').on('click', lambda: ui.notify('Play/Pause (stub)'))
            # Options
            with ui.expansion('Options').classes('w-full'):
                heikin_switch = ui.switch('Heikin Ashi', value=False)
                overlay_vol_switch = ui.switch('Overlay Volume', value=True)
                last_price_switch = ui.switch('Last Price Line', value=True)
                session_sep_switch = ui.switch('Session Separators', value=False)
                scale_select = ui.select(options=['Normal','Log','Percent'], value='Normal', label='Scale').classes('w-full').style('background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;')
                ui.button('Reset Zoom', icon='refresh').classes('bg-gray-700')\
                    .on('click', lambda: ui.run_javascript(f"(function(){{var f=document.getElementById('{iframe_id}'); if(f&&f.contentWindow&&f.contentWindow._api) f.contentWindow._api.resetZoom();}})()"))
            # Indicators
            with ui.expansion('Add Indicator').classes('w-full'):
                ind_type = ui.select(['EMA','SMA','BB','Linear Regression','VWAP','RSI','MACD'], value='EMA', label='Indicator').classes('w-full').style('background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;')
                target_select = ui.select(['Price','Pane'], value='Price', label='Target').classes('w-full').style('display:none;')
                with ui.row().classes('gap-2'):
                    p1 = ui.number('Period', value=20, min=2, max=1000, step=1).classes('w-24')
                    p2 = ui.number('Fast', value=12, min=2, max=1000, step=1).classes('w-24')
                    p3 = ui.number('Slow', value=26, min=2, max=1000, step=1).classes('w-24')
                    p4 = ui.number('Signal/Std', value=9, min=1, max=10, step=1).classes('w-28')
                with ui.row().classes('gap-2'):
                    line_width = ui.number('Width', value=2, min=1, max=5, step=1).classes('w-24')
                    color_pick = ui.color_input('Color', value='#2596be').classes('w-28')
                    add_btn = ui.button('Add', icon='add').classes('bg-emerald-700')
            with ui.expansion('Current Indicators').classes('w-full'):
                ind_list_container = ui.column().classes('gap-1')
            ui.button('Clear All Indicators', icon='delete').classes('bg-red-700 q-mt-md').on('click', lambda: (ind_list_container.clear(), ui.run_javascript("(function(){var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.clearAll && w._api.clearAll(); } })();" % json.dumps(iframe_id))))

        # Chart Area
        with ui.column().classes('q-pa-sm').style('flex: 1; min-width: 0; background: rgba(15,23,42,0.35); border: 1px solid #334155; border-radius: 10px;'):
            ui.label('TradingView Charts').classes('text-subtitle1').style('color:#e2e8f0')
            # Chart container (increase default height for better visibility)
            container = ui.element('div').style('height: 78vh; min-height: 520px; width: 100%;').classes('p-2')
            chart_dom_id = f"tv_chart_{uuid.uuid4().hex}"
            iframe_id = f"{chart_dom_id}_frame"
            with container:
                ui.html(f'<iframe id="{iframe_id}" style="height:100%; width:100%; border:0; display:block;"></iframe>', sanitize=False).style('height:100%; width:100%; display:block;')

    # Chart container (increase default height for better visibility)
    container = ui.element('div').style('height: 78vh; min-height: 520px; width: 100%;').classes('p-2')
    chart_dom_id = f"tv_chart_{uuid.uuid4().hex}"
    iframe_id = f"{chart_dom_id}_frame"
    with container:
        ui.html(f'<iframe id="{iframe_id}" style="height:100%; width:100%; border:0; display:block;"></iframe>', sanitize=False).style('height:100%; width:100%; display:block;')

    async def update_chart():
        symbol = symbol_select.value
        if not symbol or symbol not in instruments:
            ui.notify('Select a valid symbol', type='warning')
            return

        # Prepare request parameters
        try:
            start_dt = datetime.strptime(from_date.value, '%Y-%m-%d')
            end_dt = datetime.strptime(to_date.value, '%Y-%m-%d')
        except Exception:
            ui.notify('Invalid date range', type='negative')
            return

        interval = timeframe.value
        token = instruments[symbol]
        params = {
            'instrument': token,
            'from_date': start_dt.strftime('%Y-%m-%d'),
            'to_date': end_dt.strftime('%Y-%m-%d'),
            'interval': 1 if interval != 'minute' else 30,
            'unit': interval,
            'source': 'default',
        }

        try:
            resp = await fetch_api('/historical-data/Upstox', params=params)
            if not resp or resp.get('error'):
                msg = (resp.get('error', {}).get('message') if isinstance(resp, dict) else 'No response') or 'Error'
                ui.notify(f'Failed to fetch data: {msg}', type='negative')
                return
            candles = resp.get('data', [])
            if not candles:
                _show_empty(container, 'No data for selected range')
                return
        except Exception as e:
            logger.exception('Data fetch failed')
            _show_empty(container, 'Data fetch failed')
            return

        # Build DataFrame and optionally Heikin Ashi
        try:
            df = pd.DataFrame(candles)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                # Fallback if only epoch seconds present as 'time'
                if 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                else:
                    raise ValueError('timestamp missing in data')

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna().sort_values('timestamp')
            if heikin_switch.value:
                ha = _compute_heikin_ashi(df.rename(columns={'timestamp': 'ts'}).set_index('ts'))
                ha.reset_index(inplace=True)
                ha.rename(columns={'index': 'timestamp'}, inplace=True)
                df = pd.DataFrame({
                    'timestamp': df['timestamp'].values,
                    'open': ha['open'].values,
                    'high': ha['high'].values,
                    'low': ha['low'].values,
                    'close': ha['close'].values,
                    'volume': df['volume'].values,
                })
        except Exception as e:
            logger.exception('Error preparing chart data')
            _show_empty(container, 'Error preparing chart data')
            return

        # Convert to Lightweight Charts data format
        candle_data: List[Dict[str, Any]] = []
        volume_data: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            try:
                ts = pd.to_datetime(row['timestamp'])
                t = int(ts.timestamp())  # UTCTimestamp in seconds
            except Exception:
                t = str(row['timestamp'])
            o = float(row['open']); h = float(row['high']); l = float(row['low']); c = float(row['close'])
            v = float(row['volume']) if pd.notna(row['volume']) else 0.0
            candle_data.append({'time': t, 'open': o, 'high': h, 'low': l, 'close': c})
            up = c >= o
            volume_data.append({'time': t, 'value': v, 'color': 'rgba(0,200,81,0.5)' if up else 'rgba(255,68,68,0.5)'})

        # Render chart inside isolated iframe using official CDN to avoid global conflicts
        srcdoc_template = r"""
        <!DOCTYPE html><html><head><meta charset='utf-8'>
        <style>
          html,body,#wrap{height:100%;width:100%;margin:0;padding:0;background:#0a0a0a}
          #wrap{position:relative}
          #chart{position:absolute;inset:0}
          #legend{position:absolute;left:10px;top:8px;color:#e2e8f0;font:12px/1.2 ui-sans-serif,system-ui;padding:4px 6px;background:rgba(2,6,23,.45);border:1px solid rgba(148,163,184,.2);border-radius:4px;max-width:65%;}
          #legend span{display:inline-block;margin-right:8px;white-space:nowrap}
          #legend .sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px}
          #tools{position:absolute;right:10px;top:8px;display:flex;gap:6px}
          #tools button{background:rgba(2,6,23,.45);border:1px solid rgba(148,163,184,.2);color:#e2e8f0;font:12px;padding:3px 6px;border-radius:4px;cursor:pointer}
          #tools button.active{background:#0ea5e9}
        </style>
        </head><body><div id='wrap'><div id='chart'></div><div id='legend'></div><div id='tools'></div></div>
        <script>(function(){
          var urls = [
            'https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js',
            'https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js',
            'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js'
          ];
          function loadSeq(i){
            if(i>=urls.length){
              document.body.innerHTML='<div style="padding:1rem;color:#ef4444;background:#1f2937;border-radius:6px;">Failed to load Lightweight Charts from CDN.</div>';
              return;
            }
            var s=document.createElement('script');
            s.src=urls[i];
            s.onload=function(){
              // Verify API signature
              try{
                if(typeof LightweightCharts!=='object'||typeof LightweightCharts.createChart!=='function') throw new Error('Invalid API');
                var probe=LightweightCharts.createChart(document.createElement('div'));
                var ok = probe && (typeof probe.addLineSeries==='function' || typeof probe.addCandlestickSeries==='function');
                if(!ok) throw new Error('Instance lacks series methods');
                init();
              }catch(e){
                s.remove();
                loadSeq(i+1);
              }
            };
            s.onerror=function(){ s.remove(); loadSeq(i+1); };
            document.head.appendChild(s);
          }

          function init(){
            var el=document.getElementById('chart');
            var legendEl=document.getElementById('legend');
            var toolsEl=document.getElementById('tools');
            var chart=LightweightCharts.createChart(el,{ rightPriceScale:{ visible:true }, layout:{ background:{ type:'solid', color:'#0a0a0a' }, textColor:'#ddd' }, grid:{ vertLines:{ color:'rgba(100,100,100,0.1)' }, horzLines:{ color:'rgba(100,100,100,0.1)' } }, crosshair:{ mode:1 }, timeScale:{ rightOffset:6, borderVisible:true }, autoSize:true });
            var supportsCandle = (typeof chart.addCandlestickSeries==='function');
            var supportsArea = (typeof chart.addAreaSeries==='function');
            var supportsLine = (typeof chart.addLineSeries==='function');
            var supportsHist = (typeof chart.addHistogramSeries==='function');
            if(!supportsCandle && !supportsArea && !supportsLine){
              document.body.innerHTML='<div style="padding:1rem;color:#ef4444;background:#1f2937;border-radius:6px;">Chart API methods missing on instance. Please ensure the official Lightweight Charts standalone build is loaded.</div>';
              return;
            }
            var candle = supportsCandle ? chart.addCandlestickSeries({ priceLineVisible:true })
                         : (supportsArea ? chart.addAreaSeries({ priceLineVisible:true })
                         : chart.addLineSeries({ priceLineVisible:true }));
            try { candle.setData(__CANDLE_DATA__); } catch(e){
              try { candle.setData(__CANDLE_DATA__.map(function(d){ return { time:d.time, value:d.close }; })); } catch(e2){}
            }
            var volume = supportsHist ? chart.addHistogramSeries({ priceScaleId:'', priceFormat:{ type:'volume' }, priceLineVisible:false, scaleMargins:{ top:0.8, bottom:0 } })
                        : (supportsArea ? chart.addAreaSeries({ priceScaleId:'', priceLineVisible:false })
                        : (supportsLine ? chart.addLineSeries({ priceScaleId:'', priceLineVisible:false }) : null));
            if(volume){ try { volume.setData(__VOLUME_DATA__); } catch(e){ try { volume.setData(__VOLUME_DATA__.map(function(d){ return { time:d.time, value:d.value }; })); } catch(e2){} } }
            
            // --- Indicators & Toolbox API ---
            var layers = { ema20:null, ema50:null, ema200:null, sma:null, bb_up:null, bb_mid:null, bb_lo:null, vwap:null };
            var overlayMeta = {}; var paneMeta = {};
            function updateLegend(){ try{ var parts=[]; parts.push('<span><span class="sw" style="background:#94a3b8"></span>Price</span>'); for(var k in overlayMeta){ var m=overlayMeta[k]; if(m.visible) parts.push('<span><span class="sw" style="background:'+m.color+'"></span>'+m.label+'</span>'); } for(var k in paneMeta){ var m2=paneMeta[k]; if(m2.visible) parts.push('<span><span class="sw" style="background:'+m2.color+'"></span>'+m2.label+'</span>'); } legendEl.innerHTML=parts.join(' ');}catch(e){} }
            function buildTools(){ function btn(label,mode){ var b=document.createElement('button'); b.textContent=label; b.onclick=function(){ setDrawMode(mode); var bs=toolsEl.querySelectorAll('button'); bs.forEach(x=>x.classList.remove('active')); b.classList.add('active'); }; toolsEl.appendChild(b); return b; }
              btn('Pointer','pointer').classList.add('active'); btn('Trend','trend'); btn('HLine','hline'); btn('Clear','clear'); }
            var drawMode='pointer'; var drawings=[]; var firstPoint=null;
            function setDrawMode(m){ if(m==='clear'){ try{ drawings.forEach(s=>{ try{s.setData([]);}catch(e){} }); }catch(e){} drawings=[]; drawMode='pointer'; firstPoint=null; updateLegend(); return; } drawMode=m; firstPoint=null; }
            buildTools();
            chart.subscribeClick(function(param){ try{ if(drawMode==='pointer' || !param || !param.time) return; var price=null; if(param.seriesPrices){ if(param.seriesPrices.get){ price=param.seriesPrices.get(candle)||price; } if(!price){ var it=param.seriesPrices.values?param.seriesPrices.values():null; if(it){ var n=it.next(); if(!n.done) price=n.value; } } } if(price==null) return; if(drawMode==='hline'){ var s=chart.addLineSeries({ color:'#f87171', lineWidth:1, priceLineVisible:false }); s.setData(__CANDLE_DATA__.map(function(d){ return { time:d.time, value: price }; })); drawings.push(s); return; } if(drawMode==='trend'){ if(!firstPoint){ firstPoint={ time:param.time, value:price }; } else { var s=chart.addLineSeries({ color:'#fbbf24', lineWidth:2, priceLineVisible:false }); s.setData([ firstPoint, { time:param.time, value: price } ]); drawings.push(s); firstPoint=null; } } }catch(e){} });
            function computeEMA(series, period){
              var res=[], k=2/(period+1), prev=null, sum=0;
              for(var i=0;i<series.length;i++){
                var v=series[i].close;
                if(prev===null){
                  sum+=v;
                  if(i===period-1){ prev=sum/period; res.push({time:series[i].time,value:prev}); }
                } else {
                  prev=v*k+prev*(1-k);
                  res.push({time:series[i].time,value:prev});
                }
              }
              return res;
            }
            function computeSMA(series, period){
              var res=[], sum=0;
              for(var i=0;i<series.length;i++){
                var v=series[i].close; sum+=v; if(i>=period){ sum-=series[i-period].close; }
                if(i>=period-1){ res.push({time:series[i].time, value: sum/period}); }
              }
              return res;
            }
            function computeVWAP(series){
              var res=[], cumPV=0, cumV=0;
              for(var i=0;i<series.length;i++){
                var tp=(series[i].high+series[i].low+series[i].close)/3; var vol=__VOLUME_DATA__[i]?__VOLUME_DATA__[i].value:0;
                cumPV+=tp*vol; cumV+=vol; if(cumV>0){ res.push({time:series[i].time, value:cumPV/cumV}); }
              }
              return res;
            }
            function computeBB(series, period, stdev){
              var mid=computeSMA(series, period), up=[], lo=[]; var mapMid={};
              for(var i=0;i<mid.length;i++){ mapMid[mid[i].time]=mid[i].value; }
              for(var i=period-1;i<series.length;i++){
                var t=series[i].time; var mean=mapMid[t]; if(mean===undefined) continue;
                var s=0; for(var j=i-period+1;j<=i;j++){ var d=series[j].close-mean; s+=d*d; }
                var sd=Math.sqrt(s/period); up.push({time:t, value: mean+sd*stdev}); lo.push({time:t, value: mean-sd*stdev});
              }
              return {mid:mid, up:up, lo:lo};
            }
            function ensureLine(name,color){ if(!layers[name]) layers[name]=chart.addLineSeries({color:color,lineWidth:2,priceLineVisible:false}); return layers[name]; }
            function setEMAs(opts){
              var data=__CANDLE_DATA__;
              if(opts.ema20 && opts.ema20.on){ ensureLine('ema20','#f59e0b').setData(computeEMA(data, opts.ema20.period||20)); } else if(layers.ema20){ layers.ema20.setData([]); }
              if(opts.ema50 && opts.ema50.on){ ensureLine('ema50','#10b981').setData(computeEMA(data, opts.ema50.period||50)); } else if(layers.ema50){ layers.ema50.setData([]); }
              if(opts.ema200 && opts.ema200.on){ ensureLine('ema200','#6366f1').setData(computeEMA(data, opts.ema200.period||200)); } else if(layers.ema200){ layers.ema200.setData([]); }
            }
            function setSMA(opts){ var data=__CANDLE_DATA__; if(opts&&opts.on){ ensureLine('sma','#eab308').setData(computeSMA(data, opts.period||20)); } else if(layers.sma){ layers.sma.setData([]); } }
            function setBB(opts){ var data=__CANDLE_DATA__; if(opts&&opts.on){ var bb=computeBB(data, opts.period||20, opts.std||2); ensureLine('bb_up','#ef4444').setData(bb.up); ensureLine('bb_mid','#3b82f6').setData(bb.mid); ensureLine('bb_lo','#22c55e').setData(bb.lo); } else { if(layers.bb_up){layers.bb_up.setData([]);} if(layers.bb_mid){layers.bb_mid.setData([]);} if(layers.bb_lo){layers.bb_lo.setData([]);} } }
            function setVWAP(on){ if(on){ ensureLine('vwap','#f97316').setData(computeVWAP(__CANDLE_DATA__)); } else if(layers.vwap){ layers.vwap.setData([]); } }
            // Overlay & pane registries + APIs for dynamic manager
            var overlaysReg = {}; var panesReg = {};
            function setScaleMode(mode){ try{ chart.applyOptions({ rightPriceScale:{ mode: mode } }); }catch(e){} }
            function setLastPriceLine(on){ try{ if(candle && candle.applyOptions) candle.applyOptions({ priceLineVisible: !!on }); }catch(e){} }
            function setSessionSeparators(on){
              try{
                if(!candle||!candle.setMarkers){ return; }
                if(!on){ candle.setMarkers([]); return; }
                var marks=[]; var prevDay=null;
                for(var i=0;i<__CANDLE_DATA__.length;i++){
                  var t=__CANDLE_DATA__[i].time; var d=new Date(t*1000).toISOString().slice(0,10);
                  if(prevDay!==null && d!==prevDay){ marks.push({ time: __CANDLE_DATA__[i].time, position:'belowBar', color:'#475569', shape:'verticalLine', text:d }); }
                  prevDay=d;
                }
                candle.setMarkers(marks);
              }catch(e){}
            }
            function resetZoom(){ try{ chart.timeScale().fitContent(); indChart.timeScale().fitContent(); }catch(e){} }

            function addOverlay(cfg){
              var s, data, id=cfg.id;
              var c=cfg.color||'#f59e0b', w=cfg.width||2;
              function line(){ return chart.addLineSeries({color:c,lineWidth:w,priceLineVisible:false}); }
              var label='';
              if(cfg.type==='EMA'){ s=line(); data=computeEMA(__CANDLE_DATA__, cfg.params&&cfg.params.period||20); s.setData(data); overlaysReg[id]={series:[s], data:[data]}; label='EMA('+(cfg.params&&cfg.params.period||20)+')'; overlayMeta[id]={label:label,color:c,visible:true}; }
              else if(cfg.type==='SMA'){ s=line(); data=computeSMA(__CANDLE_DATA__, cfg.params&&cfg.params.period||20); s.setData(data); overlaysReg[id]={series:[s], data:[data]}; label='SMA('+(cfg.params&&cfg.params.period||20)+')'; overlayMeta[id]={label:label,color:c,visible:true}; }
              else if(cfg.type==='VWAP'){ s=line(); data=computeVWAP(__CANDLE_DATA__); s.setData(data); overlaysReg[id]={series:[s], data:[data]}; label='VWAP'; overlayMeta[id]={label:label,color:c,visible:true}; }
              else if(cfg.type==='BB'){ var bb=computeBB(__CANDLE_DATA__, cfg.params&&cfg.params.period||20, cfg.params&&cfg.params.std||2); var up=line(); up.applyOptions({color:'#ef4444'}); var mid=line(); mid.applyOptions({color:c}); var lo=line(); lo.applyOptions({color:'#22c55e'}); up.setData(bb.up); mid.setData(bb.mid); lo.setData(bb.lo); overlaysReg[id]={series:[up,mid,lo], data:[bb.up,bb.mid,bb.lo]}; label='BB('+(cfg.params&&cfg.params.period||20)+','+(cfg.params&&cfg.params.std||2)+')'; overlayMeta[id]={label:label,color:c,visible:true}; }
              else if(cfg.type==='Linear Regression'){ s=line(); var p=cfg.params&&cfg.params.period||20; var arr=[]; for(var i=p-1;i<__CANDLE_DATA__.length;i++){ var sumX=0,sumY=0,sumXY=0,sumX2=0,n=p; for(var j=0;j<p;j++){ var x=j, y=__CANDLE_DATA__[i-p+1+j].close; sumX+=x; sumY+=y; sumXY+=x*y; sumX2+=x*x;} var denom=n*sumX2 - sumX*sumX; var a=(n*sumXY - sumX*sumY)/denom; var b=(sumY - a*sumX)/n; var yhat=a*(p-1)+b; arr.push({time:__CANDLE_DATA__[i].time, value:yhat}); } s.setData(arr); overlaysReg[id]={series:[s], data:[arr]}; label='LR('+p+')'; overlayMeta[id]={label:label,color:c,visible:true}; }
              updateLegend();
            }
            function setOverlayVisible(id, vis){ var r=overlaysReg[id]; if(!r) return; for(var i=0;i<r.series.length;i++){ r.series[i].setData(vis ? r.data[i] : []); } if(overlayMeta[id]) overlayMeta[id].visible=!!vis; updateLegend(); }
            function removeOverlay(id){ var r=overlaysReg[id]; if(!r) return; for(var i=0;i<r.series.length;i++){ try{ r.series[i].setData([]); }catch(e){} } delete overlaysReg[id]; if(overlayMeta[id]) delete overlayMeta[id]; updateLegend(); }

            // Lazy pane for RSI/MACD with time sync
            var priceDiv=document.getElementById('chart');
            var paneDiv=null; var indChart=null; var syncing=false;
            function ensurePane(){ if(indChart) return; paneDiv=document.createElement('div'); paneDiv.style.cssText='height:32%;width:100%'; priceDiv.style.height='68%'; priceDiv.parentNode.appendChild(paneDiv);
              indChart=LightweightCharts.createChart(paneDiv,{ rightPriceScale:{visible:true}, layout:{ background:{ type:'solid', color:'#0a0a0a' }, textColor:'#bdbdbd' }, grid:{ vertLines:{ color:'rgba(100,100,100,0.08)' }, horzLines:{ color:'rgba(100,100,100,0.08)' } }, timeScale:{ rightOffset:6, borderVisible:true }, autoSize:true });
              chart.timeScale().subscribeVisibleTimeRangeChange(function(r){ if(syncing||!indChart) return; syncing=true; try{ indChart.timeScale().setVisibleRange(r); }finally{ syncing=false; } });
              indChart.timeScale().subscribeVisibleTimeRangeChange(function(r){ if(syncing) return; syncing=true; try{ chart.timeScale().setVisibleRange(r); }finally{ syncing=false; } });
            }
            function maybeDropPane(){ if(!indChart) return; if(Object.keys(panesReg).length===0){ try{ indChart.remove(); }catch(e){} indChart=null; try{ paneDiv.parentNode.removeChild(paneDiv);}catch(e){} paneDiv=null; priceDiv.style.height='100%'; } }

            function computeRSI(series, period){ var res=[]; var gains=0, losses=0, avgG=null, avgL=null; for(var i=1;i<series.length;i++){ var ch=series[i].close - series[i-1].close; if(i<=period){ if(ch>0) gains+=ch; else losses-=ch; if(i===period){ avgG=gains/period; avgL=losses/period; var rs=avgL===0?100:avgG/avgL; var rsi=100 - 100/(1+rs); res.push({time:series[i].time, value:rsi}); } } else { var up=ch>0?ch:0, down=ch<0? -ch:0; avgG=(avgG*(period-1)+up)/period; avgL=(avgL*(period-1)+down)/period; var rs2=avgL===0?100:avgG/avgL; var rsi2=100 - 100/(1+rs2); res.push({time:series[i].time, value:rsi2}); } } return res; }
            function computeMACD(series, fast, slow, signal){ function EMA(per){ var k=2/(per+1), out=[], prev=null; for(var i=0;i<series.length;i++){ var v=series[i].close; prev = (prev===null)? v : (v*k + prev*(1-k)); if(i>=per-1) out.push({time:series[i].time, value:prev}); } return out; } var emaF=EMA(fast), emaS=EMA(slow); var map={}, macd=[]; for(var i=0;i<emaF.length;i++){ map[emaF[i].time]=emaF[i].value; } for(var j=0;j<emaS.length;j++){ var t=emaS[j].time; if(map[t]!==undefined){ macd.push({time:t, value: map[t]-emaS[j].value}); } } var sig=[]; var k=2/(signal+1), prev=null; for(var i=0;i<macd.length;i++){ var v=macd[i].value; prev = (prev===null)? v : (v*k + prev*(1-k)); sig.push({time:macd[i].time, value: prev}); } var hist=[]; var mapSig={}; for(var i=0;i<sig.length;i++){ mapSig[sig[i].time]=sig[i].value; } for(var i=0;i<macd.length;i++){ var t2=macd[i].time; if(mapSig[t2]!==undefined){ hist.push({time:t2, value: macd[i].value - mapSig[t2]}); } } return {macd:macd, signal:sig, hist:hist}; }

            function addPaneIndicator(cfg){ var id=cfg.id; ensurePane(); if(cfg.type==='RSI'){ var r=indChart.addLineSeries({color:cfg.color||'#3b82f6', lineWidth:cfg.width||2, priceLineVisible:false}); var data=computeRSI(__CANDLE_DATA__, cfg.params&&cfg.params.period||14); r.setData(data); panesReg[id]={series:[r], data:[data]}; paneMeta[id]={label:'RSI('+(cfg.params&&cfg.params.period||14)+')', color:cfg.color||'#3b82f6', visible:true}; updateLegend(); }
              else if(cfg.type==='MACD'){ var m=computeMACD(__CANDLE_DATA__, cfg.params&&cfg.params.fast||12, cfg.params&&cfg.params.slow||26, cfg.params&&cfg.params.signal||9); var l1=indChart.addLineSeries({color:'#22c55e', lineWidth:2, priceLineVisible:false}); var l2=indChart.addLineSeries({color:'#ef4444', lineWidth:2, priceLineVisible:false}); var h=indChart.addHistogramSeries({ priceFormat:{type:'volume'}, priceLineVisible:false, color:'#9ca3af' }); l1.setData(m.macd); l2.setData(m.signal); h.setData(m.hist); panesReg[id]={series:[l1,l2,h], data:[m.macd, m.signal, m.hist]}; paneMeta[id]={label:'MACD('+(cfg.params&&cfg.params.fast||12)+','+(cfg.params&&cfg.params.slow||26)+','+(cfg.params&&cfg.params.signal||9)+')', color:'#22c55e', visible:true}; updateLegend(); }
            }
            function setPaneVisible(id, vis){ var r=panesReg[id]; if(!r) return; for(var i=0;i<r.series.length;i++){ r.series[i].setData(vis ? r.data[i] : []); } if(paneMeta[id]) paneMeta[id].visible=!!vis; updateLegend(); maybeDropPane(); }
            function removePaneIndicator(id){ var r=panesReg[id]; if(!r) return; for(var i=0;i<r.series.length;i++){ try{ r.series[i].setData([]);}catch(e){} } delete panesReg[id]; if(paneMeta[id]) delete paneMeta[id]; updateLegend(); maybeDropPane(); }

            function clearAll(){ try{ for(var id in overlaysReg){ removeOverlay(id);} for(var id2 in panesReg){ removePaneIndicator(id2);} if(candle&&candle.setMarkers) candle.setMarkers([]);}catch(e){} }
            window._api = { setEMAs:setEMAs, setSMA:setSMA, setBB:setBB, setVWAP:setVWAP, setScaleMode:setScaleMode, setLastPriceLine:setLastPriceLine, setSessionSeparators:setSessionSeparators, resetZoom:resetZoom, addOverlay:addOverlay, setOverlayVisible:setOverlayVisible, removeOverlay:removeOverlay, addPaneIndicator:addPaneIndicator, setPaneVisible:setPaneVisible, removePaneIndicator:removePaneIndicator, clearAll:clearAll };
            window.addEventListener('resize', function(){ try{ chart.resize(el.clientWidth, el.clientHeight); }catch(e){} });
          }

          loadSeq(0);
        })();</script>
        </body></html>
        """
        srcdoc = (
            srcdoc_template
            .replace('__CANDLE_DATA__', json.dumps(candle_data))
            .replace('__VOLUME_DATA__', json.dumps(volume_data))
        )
        # Build one-shot JS to set iframe srcdoc and apply initial indicator states
        initial_ema = { 'ema20': {'on': False, 'period': 20}, 'ema50': {'on': False, 'period': 50}, 'ema200': {'on': False, 'period': 200} }
        initial_sma = {'on': False, 'period': 20}
        initial_bb = {'on': False, 'period': 20, 'std': 2.0}
        initial_vwap = False
        scale_map = {'Normal':0, 'Log':1, 'Percent':2}
        scale_mode = scale_map.get(scale_select.value, 0)

        js = (
            "(function(){var f=document.getElementById(%s); if(!f) return; var html=%s; f.srcdoc=JSON.parse(html);"
            "setTimeout(function(){ try{ var w=f.contentWindow; if(w && w._api){ w._api.setEMAs(%s); w._api.setSMA(%s); w._api.setBB(%s); w._api.setVWAP(%s); w._api.setScaleMode(%s); w._api.setLastPriceLine(%s); w._api.setSessionSeparators(%s); } }catch(e){} }, 900); })();"
            % (json.dumps(iframe_id), json.dumps(json.dumps(srcdoc)), json.dumps(initial_ema), json.dumps(initial_sma), json.dumps(initial_bb), json.dumps(initial_vwap), json.dumps(scale_mode), json.dumps(True), json.dumps(False))
        )
        ui.run_javascript(js)

        # Restore saved layout per symbol, if any
        try:
            current_symbol = symbol_select.value
            layouts = user_storage.get('tv_layouts', {}) or {}
            saved = layouts.get(current_symbol, [])
            if saved:
                def apply_saved():
                    for cfg in saved:
                        cfg_js = json.dumps(cfg)
                        if cfg.get('target') == 'Pane' and cfg.get('type') in ('RSI','MACD'):
                            call = "(function(){var tries=0; function go(){ var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.addPaneIndicator(%s); %s } else if(tries++<20){ setTimeout(go,120);} } go(); })();"
                        else:
                            call = "(function(){var tries=0; function go(){ var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.addOverlay(%s); %s } else if(tries++<20){ setTimeout(go,120);} } go(); })();"
                        vis = cfg.get('visible', True)
                        post = "" if vis else ("var id="+json.dumps(cfg.get('id'))+"; if(w._api.setOverlayVisible) w._api.setOverlayVisible(id,false); if(w._api.setPaneVisible) w._api.setPaneVisible(id,false);")
                        ui.run_javascript(call % (json.dumps(iframe_id), cfg_js, post))
                ui.timer(1.2, apply_saved, once=True)
        except Exception:
            pass

        # Wire up handlers for toolbox changes
        def apply_toolbox():
            # Apply scale only (legacy switches removed in favor of manager)
            scale_map = { 'Normal':0, 'Log':1, 'Percent':2 }
            scale_mode = scale_map.get(scale_select.value, 0)
            ui.run_javascript(
                "(function(){var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w && w._api){ w._api.setScaleMode(%s); } })();"
                % (json.dumps(iframe_id), json.dumps(scale_mode))
            )

        scale_select.on('update:model-value', lambda: apply_toolbox())
        # Apply last price line and session separators toggles
        last_price_switch.on('update:model-value', lambda: ui.run_javascript(
            "(function(){var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.setLastPriceLine(%s); } })();"
            % (json.dumps(iframe_id), json.dumps(True)) if last_price_switch.value else
            "(function(){var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.setLastPriceLine(%s); } })();"
            % (json.dumps(iframe_id), json.dumps(False))
        ))
        session_sep_switch.on('update:model-value', lambda: ui.run_javascript(
            "(function(){var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.setSessionSeparators(%s); } })();"
            % (json.dumps(iframe_id), json.dumps(True)) if session_sep_switch.value else
            "(function(){var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.setSessionSeparators(%s); } })();"
            % (json.dumps(iframe_id), json.dumps(False))
        ))

        # Indicator Manager actions
        def sync_indicator_inputs():
            # Auto target + dynamic inputs
            itype = ind_type.value
            if itype in ('RSI','MACD'):
                target_select.value = 'Pane'
            else:
                target_select.value = 'Price'
            # Show/hide inputs
            if itype in ('EMA','SMA','RSI','Linear Regression'):
                p1.style('')
                p2.style('display:none')
                p3.style('display:none')
                p4.style('display:none')
            elif itype == 'BB':
                p1.style('')
                p2.style('display:none')
                p3.style('display:none')
                p4.label = 'Std'
                p4.style('')
            elif itype == 'MACD':
                p1.style('display:none')
                p2.label = 'Fast'; p2.style('')
                p3.label = 'Slow'; p3.style('')
                p4.label = 'Signal'; p4.style('')
            elif itype == 'VWAP':
                p1.style('display:none'); p2.style('display:none'); p3.style('display:none'); p4.style('display:none')

        ind_type.on('update:model-value', lambda: sync_indicator_inputs())
        # Initialize fields
        sync_indicator_inputs()

        def add_indicator():
            itype = ind_type.value
            # Auto target selection
            target = 'Pane' if itype in ('RSI','MACD') else 'Price'
            cfg = {
                'id': str(uuid.uuid4()),
                'type': itype,
                'target': target,
                'params': {},
                'color': color_pick.value or '#f59e0b',
                'width': int(line_width.value) if line_width.value else 2,
                'visible': True,
            }
            # Assign params depending on indicator
            if itype in ('EMA','SMA','RSI','Linear Regression'):
                cfg['params']['period'] = int(p1.value) if p1.value else 20
            if itype == 'MACD':
                cfg['params']['fast'] = int(p2.value) if p2.value else 12
                cfg['params']['slow'] = int(p3.value) if p3.value else 26
                cfg['params']['signal'] = int(p4.value) if p4.value else 9
            if itype == 'BB':
                cfg['params']['period'] = int(p1.value) if p1.value else 20
                cfg['params']['std'] = float(p4.value) if p4.value else 2.0

            # Wait for _api and then call
            if target == 'Pane' and itype in ('RSI','MACD'):
                js_call = "(function(){var tries=0; function go(){ var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.addPaneIndicator(%s); } else if(tries++<20){ setTimeout(go,100);} } go(); })();" % (json.dumps(iframe_id), json.dumps(cfg))
            else:
                js_call = "(function(){var tries=0; function go(){ var f=document.getElementById(%s); var w=f&&f.contentWindow; if(w&&w._api){ w._api.addOverlay(%s); } else if(tries++<20){ setTimeout(go,100);} } go(); })();" % (json.dumps(iframe_id), json.dumps(cfg))
            ui.run_javascript(js_call)

            # Persist per-symbol layout
            try:
                sym = symbol_select.value
                layouts = user_storage.get('tv_layouts', {}) or {}
                lst = list(layouts.get(sym, []))
                lst.append(cfg)
                layouts[sym] = lst
                user_storage['tv_layouts'] = layouts
            except Exception:
                pass

            # Add to list UI
            with ind_list_container:
                row_id = cfg['id']
                with ui.row().classes('items-center gap-2').style('background: rgba(30,41,59,0.4); padding: 6px; border-radius: 6px;') as r:
                    ui.label(f"{cfg['type']} [{cfg['params']}] -> {cfg['target']}").classes('text-sm')
                    vis = ui.switch('Show', value=True)
                    def toggle_visibility(e=None, _id=row_id, _target=cfg['target']):
                        js = "(function(){var f=document.getElementById(%s); var w=f&&f.contentWindow; if(!w||!w._api) return; %s })();"
                        if _target == 'Pane' and cfg['type'] in ('RSI','MACD'):
                            action = "w._api.setPaneVisible(%s,%s);" % (json.dumps(_id), json.dumps(bool(vis.value)))
                        else:
                            action = "w._api.setOverlayVisible(%s,%s);" % (json.dumps(_id), json.dumps(bool(vis.value)))
                        ui.run_javascript(js % (json.dumps(iframe_id), action))
                        # persist visibility
                        try:
                            sym = symbol_select.value
                            layouts = user_storage.get('tv_layouts', {}) or {}
                            lst = list(layouts.get(sym, []))
                            for item in lst:
                                if item.get('id') == _id:
                                    item['visible'] = bool(vis.value)
                                    break
                            layouts[sym] = lst
                            user_storage['tv_layouts'] = layouts
                        except Exception:
                            pass
                    vis.on('update:model-value', toggle_visibility)
                    def remove_indicator(_id=row_id, _target=cfg['target'], _row=r, _type=cfg['type']):
                        js = "(function(){var f=document.getElementById(%s); var w=f&&f.contentWindow; if(!w||!w._api) return; %s })();"
                        if _target == 'Pane' and _type in ('RSI','MACD'):
                            action = "w._api.removePaneIndicator(%s);" % json.dumps(_id)
                        else:
                            action = "w._api.removeOverlay(%s);" % json.dumps(_id)
                        ui.run_javascript(js % (json.dumps(iframe_id), action))
                        _row.delete()
                        # persist removal
                        try:
                            sym = symbol_select.value
                            layouts = user_storage.get('tv_layouts', {}) or {}
                            lst = [item for item in layouts.get(sym, []) if item.get('id') != _id]
                            layouts[sym] = lst
                            user_storage['tv_layouts'] = layouts
                        except Exception:
                            pass
                    ui.button('Remove', icon='close').classes('bg-red-700').on('click', remove_indicator)

        add_btn.on('click', add_indicator)

    def _show_empty(parent, message: str):
        parent.clear()
        with parent:
            ui.html(f'<div style="height:100%;display:flex;align-items:center;justify-content:center;color:#94a3b8;">{message}</div>')

    # Initial render (slightly delayed to allow script load)
    ui.timer(1.5, lambda: update_chart(), once=True)
