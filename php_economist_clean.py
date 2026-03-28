#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "yfinance>=0.2.52",
#   "pandas>=2.2",
#   "numpy>=1.26",
#   "matplotlib>=3.9",
#   "statsmodels>=0.14",
# ]
# ///
"""
Philippine Peso Depreciation --- Economist-Styled Chart Suite

Design system: The Economist (Los Angeles 95, Economist Red, Chicago Blue)
Usage:
    uv run php_economist_clean.py
    uv run php_economist_clean.py --out-dir ./images

Output PNGs are written to OUT_DIR (default: ./images).
"""

import os
import sys
import warnings
import argparse
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# ECONOMIST PALETTE
BG        = '#F5F4EF'   # Los Angeles 95 --- canvas
RED       = '#E3120B'   # Economist Red
BLUE      = '#2E45B8'   # Chicago 45
TEAL      = '#1DC9A4'   # Hong Kong 45
GREY_TEXT = '#595959'   # London 35 --- source, captions
GREY_RULE = '#D9D9D9'   # London 85 --- grid lines
GREY_DARK = '#333333'   # London 20 --- axis ticks
BLACK     = '#0D0D0D'   # London 5  --- titles
WHITE     = '#FFFFFF'
CANVAS_DARK = '#E1DFD0' # Los Angeles 85 --- shaded regions

TRUMP_DATE  = pd.Timestamp('2025-01-20')
START_DATE  = '2018-01-01'
TRAIN_START = '2020-07-01'
END_DATE    = datetime.today().strftime('%Y-%m-%d')
USER_HYPO   = 70.0

# CONFIG & OUTPUT
parser = argparse.ArgumentParser(description='Generate Economist-styled PHP depreciation charts.')
parser.add_argument('--out-dir', default=os.environ.get('OUT_DIR', './images'),
                    help='Output directory (default: ./images)')
args, _ = parser.parse_known_args()
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# TYPOGRAPHY
mpl.rcParams.update({
    'font.family'         : 'serif',
    'font.serif'          : ['Lora', 'TeX Gyre Pagella', 'Liberation Serif', 'DejaVu Serif'],
    'font.sans-serif'     : ['Poppins', 'Liberation Sans', 'DejaVu Sans'],
    'font.size'           : 10,
    'axes.facecolor'      : BG,
    'figure.facecolor'    : BG,
    'axes.edgecolor'      : GREY_RULE,
    'axes.labelcolor'     : GREY_DARK,
    'xtick.color'         : GREY_DARK,
    'ytick.color'         : GREY_DARK,
    'xtick.labelsize'     : 8.5,
    'ytick.labelsize'     : 8.5,
    'axes.titlecolor'     : BLACK,
    'text.color'          : BLACK,
    'grid.color'          : GREY_RULE,
    'grid.linewidth'      : 0.65,
    'grid.linestyle'      : '-',
    'axes.spines.top'     : False,
    'axes.spines.right'   : False,
    'axes.spines.left'    : False,
    'axes.spines.bottom'  : True,
    'axes.linewidth'      : 0.6,
    'xtick.bottom'        : True,
    'ytick.left'          : False,
    'xtick.major.size'    : 3,
    'legend.frameon'      : False,
    'legend.fontsize'     : 8.5,
    'legend.labelcolor'   : GREY_DARK,
})

# DATA
def safe_dl(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        c = raw['Close'].iloc[:, 0]
    else:
        c = raw['Close']
    return pd.Series(c.values.flatten(), index=c.index, name=ticker).dropna()

print('Fetching data  ')
php = safe_dl('USDPHP=X', START_DATE, END_DATE)
dxy = safe_dl('DX-Y.NYB', START_DATE, END_DATE)
df  = pd.DataFrame({'USDPHP': php, 'DXY': dxy}).dropna().sort_index()

pre  = df[df.index < TRUMP_DATE]
post = df[df.index >= TRUMP_DATE].copy()

dxy_base    = post['DXY'].iloc[0]
php_inaug   = post['USDPHP'].iloc[0]
php_now     = post['USDPHP'].iloc[-1]
dxy_now     = post['DXY'].iloc[-1]
dxy_chg_pct = (dxy_now - dxy_base) / dxy_base * 100
today_str   = df.index[-1].strftime('%d %b %Y')

# ARIMA
train = pre.loc[TRAIN_START:, 'USDPHP']
arima_fit = ARIMA(train.values, order=(1,1,1), trend='t').fit(
    method_kwargs={'warn_convergence': False})
fc       = arima_fit.get_forecast(steps=len(post))
fc_mean  = pd.Series(fc.predicted_mean,            index=post.index, name='ARIMA')
fc_lo    = pd.Series(fc.conf_int(alpha=0.10)[:,0], index=post.index)
fc_hi    = pd.Series(fc.conf_int(alpha=0.10)[:,1], index=post.index)

# DXY-normalization
post['USDPHP_adj'] = post['USDPHP'] * (dxy_base / post['DXY'])
gap = post['USDPHP_adj'] - post['USDPHP']

adj_now   = post['USDPHP_adj'].iloc[-1]
arima_now = fc_mean.iloc[-1]
gap_now   = gap.iloc[-1]

print(f'  USDPHP today   : {php_now:.4f}')
print(f'  DXY-Adj today  : {adj_now:.4f}')
print(f'  ARIMA today    : {arima_now:.4f}')
print(f'  DXY change     : {dxy_chg_pct:+.2f}%')

# ECONOMIST CHART FRAME
def economist_figure(figw=10, figh=6):
    """Return fig, ax with Economist chrome --- red top rule, clean axes."""
    fig, ax = plt.subplots(figsize=(figw, figh), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.14)
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GREY_RULE, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_tick_params(length=4, color=GREY_RULE)
    ax.spines['bottom'].set_color(GREY_DARK)
    ax.spines['bottom'].set_linewidth(0.6)
    return fig, ax

def add_economist_chrome(fig, ax, title, subtitle, source,
                         red_rule_y=0.915, rule_height=0.012):
    """Add Economist red top rule, bold title, subtitle, and source line."""
    fig.add_artist(
        mpl.patches.FancyBboxPatch(
            (0.0, red_rule_y), 1.0, rule_height,
            boxstyle='square,pad=0', linewidth=0,
            facecolor=RED, transform=fig.transFigure, zorder=10, clip_on=False
        )
    )
    fig.text(0.08, red_rule_y - 0.02, title,
             fontsize=13.5, fontweight='bold', color=BLACK,
             va='top', transform=fig.transFigure,
             fontfamily='Lora')
    fig.text(0.08, red_rule_y - 0.075, subtitle,
             fontsize=9, color=GREY_TEXT,
             va='top', transform=fig.transFigure,
             fontfamily='Lora', style='italic')
    fig.text(0.08, 0.025, f'Sources: {source}',
             fontsize=7.5, color=GREY_TEXT,
             va='bottom', transform=fig.transFigure,
             fontfamily='Lora')
    fig.text(0.96, 0.025, f'As of {today_str}',
             fontsize=7.5, color=GREY_TEXT, ha='right',
             va='bottom', transform=fig.transFigure,
             fontfamily='Lora')

def fmt_date_axis(ax, full=True):
    if full:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

def inline_label(ax, x, y, text, color, xoffset_days=10, yoffset=0, fs=8.2):
    ax.annotate(
        text, xy=(x, y),
        xytext=(x + pd.Timedelta(days=xoffset_days), y + yoffset),
        color=color, fontsize=fs, fontweight='bold',
        fontfamily='Lora',
        arrowprops=None,
        bbox=dict(boxstyle='round,pad=0.2', facecolor=BG,
                  edgecolor='none', alpha=0.85)
    )

def trump_vline(ax, ymin, ymax):
    ax.axvline(TRUMP_DATE, color=GREY_DARK, lw=0.9, ls=':', zorder=2)
    ax.axvspan(TRUMP_DATE, df.index[-1], color=CANVAS_DARK, alpha=0.35, zorder=0)
    mid_y = ymin + (ymax - ymin) * 0.06
    ax.text(TRUMP_DATE + pd.Timedelta(days=8), mid_y,
            'Trump\ninauguration\n20 Jan 2025',
            color=GREY_TEXT, fontsize=7.2, va='bottom',
            fontfamily='Lora', style='italic')

# CHART 1 - Full History + Both Counterfactuals
print('\nChart 1  ')
fig1, ax1 = economist_figure(10, 6)

# Pre-Trump actual
ax1.plot(pre.index, pre['USDPHP'], color=RED, lw=1.6, zorder=4)
# Post-Trump actual (thicker)
ax1.plot(post.index, post['USDPHP'], color=RED, lw=2.2, zorder=4)
# ARIMA CI band
ax1.fill_between(fc_lo.index, fc_lo.values, fc_hi.values,
                  color=BLUE, alpha=0.10, zorder=1)
# ARIMA mean
ax1.plot(fc_mean.index, fc_mean.values,
         color=BLUE, lw=1.9, ls='--', zorder=3, dashes=(5,3))
# DXY-adjusted
ax1.plot(post.index, post['USDPHP_adj'],
         color=TEAL, lw=1.9, ls='--', zorder=3, dashes=(3,2))
# Hypothesis
ax1.axhline(USER_HYPO, color=GREY_DARK, lw=1.0, ls=':', alpha=0.6, zorder=2)

ylo = df['USDPHP'].min() * 0.983
yhi = max(post['USDPHP_adj'].max(), fc_hi.max(), USER_HYPO) * 1.018
ax1.set_ylim(ylo, yhi)

trump_vline(ax1, ylo, yhi)

# End-point inline labels
inline_label(ax1, post.index[-1], php_now,
             f'Actual: {php_now:.2f}', RED, xoffset_days=7, yoffset=-0.3)
inline_label(ax1, post.index[-1], adj_now,
             f'DXY-adj: {adj_now:.2f}', TEAL, xoffset_days=7, yoffset=0.4)
inline_label(ax1, fc_mean.index[-1], arima_now,
             f'ARIMA: {arima_now:.2f}', BLUE, xoffset_days=7, yoffset=-0.8)
ax1.text(df.index[-1] - pd.Timedelta(days=80), USER_HYPO + 0.15,
         f'Hypothesis: {USER_HYPO:.0f}',
         color=GREY_DARK, fontsize=7.5, fontfamily='Lora', style='italic')

ax1.set_ylabel('Philippine pesos per US dollar', fontsize=8.5, color=GREY_TEXT,
               fontfamily='Lora')
fmt_date_axis(ax1, full=True)

# Compact legend
legend_elements = [
    Line2D([0],[0], color=RED,  lw=2,   label='Actual USDPHP'),
    Line2D([0],[0], color=BLUE, lw=1.8, ls='--', dashes=(5,3), label='ARIMA counterfactual (pre-Trump trend, 90% CI)'),
    Line2D([0],[0], color=TEAL, lw=1.8, ls='--', dashes=(3,2), label='DXY-normalised (USD weakness stripped out)'),
    Line2D([0],[0], color=GREY_DARK, lw=1, ls=':', label=f'Hypothesis: ~{USER_HYPO:.0f} PHP/USD'),
]
ax1.legend(handles=legend_elements, loc='upper left',
           fontsize=7.8, ncol=2, handlelength=2.4)

add_economist_chrome(
    fig1, ax1,
    title    = 'The peso is weaker than it looks',
    subtitle = 'Philippine peso per US dollar, with stable-dollar counterfactuals',
    source   = 'Yahoo Finance (USDPHP=X, DX-Y.NYB); ARIMA(1,1,1) with drift'
)
fig1.savefig(f'{OUT_DIR}/chart_1_full_history.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close(fig1)
print('  -> chart_1_full_history.png')

# CHART 2 - DXY Dollar Index
print('Chart 2  ')
fig2, ax2 = economist_figure(10, 6)

ax2.plot(df.index, df['DXY'], color=BLUE, lw=1.8, zorder=4)
ax2.axhline(dxy_base, color=GREY_DARK, lw=0.9, ls=':', alpha=0.5, zorder=2)

# Fill: weakness zone (post-Trump, DXY below baseline)
post_dxy = post['DXY']
ax2.fill_between(post_dxy.index, dxy_base, post_dxy.values,
                  where=(post_dxy.values < dxy_base),
                  color=RED, alpha=0.18, zorder=1, label=f'USD weakness (DXY below {dxy_base:.0f})')
ax2.fill_between(post_dxy.index, dxy_base, post_dxy.values,
                  where=(post_dxy.values >= dxy_base),
                  color=TEAL, alpha=0.18, zorder=1, label='USD strength')

ylo2 = df['DXY'].min() * 0.985
yhi2 = df['DXY'].max() * 1.015
ax2.set_ylim(ylo2, yhi2)
trump_vline(ax2, ylo2, yhi2)

ax2.text(df.index[-1] - pd.Timedelta(days=75), dxy_base - 0.5,
         f'Inauguration: {dxy_base:.1f}',
         color=GREY_TEXT, fontsize=7.5, fontfamily='Lora', style='italic', ha='right')

inline_label(ax2, post.index[-1], dxy_now,
             f'{dxy_now:.1f}\n({dxy_chg_pct:+.1f}%)', BLUE,
             xoffset_days=7, yoffset=0)

ax2.set_ylabel('DXY Index level', fontsize=8.5, color=GREY_TEXT, fontfamily='Lora')
fmt_date_axis(ax2, full=True)

legend_elements2 = [
    mpl.patches.Patch(color=RED,  alpha=0.5, label=f'USD weakness --- DXY below {dxy_base:.0f}'),
    mpl.patches.Patch(color=TEAL, alpha=0.5, label='USD strength'),
]
ax2.legend(handles=legend_elements2, loc='lower left', fontsize=8)

add_economist_chrome(
    fig2, ax2,
    title    = "The dollar slipped --- and took the peso's pain with it",
    subtitle = f'US Dollar Index (DXY), January 2018 to {today_str}',
    source   = 'Yahoo Finance (DX-Y.NYB); ICE Futures'
)
fig2.savefig(f'{OUT_DIR}/chart_2_dxy_dollar_index.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print('  -> chart_2_dxy_dollar_index.png')

# CHART 3 - Post-Trump Zoom
print('Chart 3  ')
fig3, ax3 = economist_figure(10, 6)

ax3.fill_between(fc_lo.index, fc_lo.values, fc_hi.values,
                  color=BLUE, alpha=0.10, zorder=1)
ax3.plot(post.index, post['USDPHP'],      color=RED,      lw=2.2, zorder=4)
ax3.plot(fc_mean.index, fc_mean.values,   color=BLUE,     lw=1.9, ls='--', dashes=(5,3), zorder=3)
ax3.plot(post.index, post['USDPHP_adj'],  color=TEAL,     lw=1.9, ls='--', dashes=(3,2), zorder=3)
ax3.axhline(USER_HYPO, color=GREY_DARK, lw=1.0, ls=':', alpha=0.55, zorder=2)

ylo3 = min(post['USDPHP'].min(), fc_lo.min()) * 0.992
yhi3 = max(post['USDPHP_adj'].max(), fc_hi.max(), USER_HYPO) * 1.012
ax3.set_ylim(ylo3, yhi3)

ax3.axvline(TRUMP_DATE, color=GREY_DARK, lw=0.9, ls=':', zorder=2)
ax3.text(TRUMP_DATE + pd.Timedelta(days=4),
         ylo3 + (yhi3 - ylo3) * 0.04,
         '20 Jan 2025', color=GREY_TEXT, fontsize=7.2,
         fontfamily='Lora', style='italic')

# Direct labels at far right
end_x = post.index[-1]
inline_label(ax3, end_x, php_now,  f'Actual: {php_now:.2f}',  RED,  xoffset_days=5, yoffset=-0.25)
inline_label(ax3, end_x, adj_now,  f'DXY-adj: {adj_now:.2f}', TEAL, xoffset_days=5, yoffset=0.35)
inline_label(ax3, end_x, arima_now, f'ARIMA: {arima_now:.2f}', BLUE, xoffset_days=5, yoffset=-0.7)
ax3.text(post.index[-1] - pd.Timedelta(days=50), USER_HYPO + 0.12,
         f'Hypothesis: {USER_HYPO:.0f}',
         color=GREY_DARK, fontsize=7.5, fontfamily='Lora', style='italic')

ax3.set_ylabel('Philippine pesos per US dollar', fontsize=8.5,
               color=GREY_TEXT, fontfamily='Lora')
fmt_date_axis(ax3, full=False)

legend_elements3 = [
    Line2D([0],[0], color=RED,  lw=2,                     label='Actual USDPHP'),
    Line2D([0],[0], color=BLUE, lw=1.8, ls='--', dashes=(5,3), label='ARIMA counterfactual (90% CI band)'),
    Line2D([0],[0], color=TEAL, lw=1.8, ls='--', dashes=(3,2), label='DXY-normalised rate'),
]
ax3.legend(handles=legend_elements3, loc='upper left', fontsize=7.8, handlelength=2.4)

add_economist_chrome(
    fig3, ax3,
    title    = 'Under Trump, the peso looks stable. It is not.',
    subtitle = f'Philippine peso per US dollar, 20 January 2025 to {today_str}',
    source   = 'Yahoo Finance (USDPHP=X); BSP reference'
)
fig3.savefig(f'{OUT_DIR}/chart_3_post_trump_zoom.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close(fig3)
print('  -> chart_3_post_trump_zoom.png')

# CHART 4 - PHP Cushion (Gap)
print('Chart 4  ')
fig4, ax4 = economist_figure(10, 6)

ax4.fill_between(gap.index, 0, gap.values,
                  where=(gap.values >= 0), color=TEAL, alpha=0.45,
                  zorder=2, label='USD weakness masking PHP decline')
ax4.fill_between(gap.index, 0, gap.values,
                  where=(gap.values < 0),  color=RED,  alpha=0.45,
                  zorder=2, label='PHP weaker even after USD adjustment')
ax4.plot(gap.index, gap.values, color=GREY_DARK, lw=0.8, alpha=0.6, zorder=3)
ax4.axhline(0, color=BLACK, lw=0.9, zorder=4)

ylo4 = gap.min() * 1.25
yhi4 = gap.max() * 1.35
ax4.set_ylim(ylo4, yhi4)

# Annotate current gap
ax4.annotate(
    f'+{gap_now:.2f} PHP/USD\ncushion today',
    xy=(gap.index[-1], gap_now),
    xytext=(gap.index[-1] - pd.Timedelta(days=55), gap_now + gap.max()*0.3),
    color=TEAL, fontsize=8.5, fontweight='bold', fontfamily='Lora',
    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.0)
)

# Area annotation
mid_x = post.index[len(post)//2]
ax4.text(mid_x, gap.max() * 0.55,
         'This area represents how much stronger the peso\nappears because of dollar weakness, not its own merits.',
         ha='center', color=GREY_TEXT, fontsize=7.8,
         fontfamily='Lora', style='italic')

ax4.set_ylabel('PHP per USD differential\n(DXY-adjusted minus actual)', fontsize=8.5,
               color=GREY_TEXT, fontfamily='Lora')
fmt_date_axis(ax4, full=False)

legend_elements4 = [
    mpl.patches.Patch(color=TEAL, alpha=0.55, label='Cushion --- USD weakness masking PHP decline'),
    mpl.patches.Patch(color=RED,  alpha=0.55, label='PHP weaker than adjusted rate'),
]
ax4.legend(handles=legend_elements4, loc='upper left', fontsize=8)

add_economist_chrome(
    fig4, ax4,
    title    = "A borrowed shield: the peso's fake strength",
    subtitle = 'DXY-normalised USDPHP minus actual USDPHP --- positive = USD weakness doing the heavy lifting',
    source   = 'Yahoo Finance (USDPHP=X, DX-Y.NYB); author calculations'
)
fig4.savefig(f'{OUT_DIR}/chart_4_php_cushion.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close(fig4)
print('  -> chart_4_php_cushion.png')

# CHART 5 - Summary Table (Economist data-table style)
print('Chart 5  ')

php_pct_actual = (php_now   - php_inaug) / php_inaug * 100
php_pct_adj    = (adj_now   - php_inaug) / php_inaug * 100
php_pct_arima  = (arima_now - php_inaug) / php_inaug * 100

table_data = [
    ('DXY at inauguration',            f'{dxy_base:.2f}',          'USD strength baseline'),
    ('DXY today',                       f'{dxy_now:.2f}',           f'{dxy_chg_pct:+.1f}% --- dollar weakened'),
    ('USDPHP at inauguration',          f'{php_inaug:.4f}',         'PHP reference rate, 20 Jan 2025'),
    ('USDPHP today (actual)',           f'{php_now:.4f}',           f'{php_pct_actual:+.2f}% vs inauguration'),
    ('USDPHP DXY-normalised',           f'{adj_now:.4f}',           f'{php_pct_adj:+.2f}% implied depreciation'),
    ('USDPHP ARIMA forecast',           f'{arima_now:.4f}',         f'{php_pct_arima:+.2f}% pre-Trump trend'),
    ('User hypothesis (stable USD)',    f'~{USER_HYPO:.1f}',        'Overshoots by ~5 PHP/USD'),
    ('PHP cushion from USD weakness',  f'+{gap_now:.4f} PHP/USD',  'Dollar drop masking PHP losses'),
]

fig5 = plt.figure(figsize=(11, 5.8), facecolor=BG)
fig5.subplots_adjust(left=0.0, right=1.0, top=0.82, bottom=0.06)

ax5 = fig5.add_subplot(111)
ax5.set_facecolor(BG)
ax5.axis('off')

# Column headers
COL_X    = [0.03, 0.38, 0.58]
HEADER_Y = 0.96
ROW_H    = 0.088

headers = ['Metric', 'Value', 'Interpretation']
for i, h in enumerate(headers):
    ax5.text(COL_X[i], HEADER_Y, h,
             transform=ax5.transAxes,
             fontsize=8.5, fontweight='bold', color=WHITE,
             fontfamily='Lora', va='center')

# Header background bar
ax5.add_patch(mpl.patches.FancyBboxPatch(
    (0.0, HEADER_Y - 0.045), 1.0, ROW_H * 0.92,
    boxstyle='square,pad=0', linewidth=0,
    facecolor=BLACK, transform=ax5.transAxes, zorder=0, clip_on=False
))

# Rows
for i, (metric, value, interp) in enumerate(table_data):
    y = HEADER_Y - ROW_H * (i + 1) - 0.01
    row_bg = CANVAS_DARK if i % 2 == 0 else BG

    ax5.add_patch(mpl.patches.FancyBboxPatch(
        (0.0, y - 0.035), 1.0, ROW_H * 0.92,
        boxstyle='square,pad=0', linewidth=0,
        facecolor=row_bg, transform=ax5.transAxes, zorder=0, clip_on=False
    ))

    ax5.text(COL_X[0], y, metric, transform=ax5.transAxes,
             fontsize=8.5, color=BLACK, fontfamily='Lora', va='center')

    # Value --- colored if it contains a sign
    val_color = RED if ('+' in value and 'cushion' in metric.lower()) or \
                       ('pct' in metric.lower() and '-' in value) else \
                TEAL if '+' in value and 'cushion' not in metric.lower() else \
                BLUE
    val_color = BLACK  # override: clean table, color only key rows
    if metric in ('USDPHP DXY-normalised', 'PHP cushion from USD weakness'):
        val_color = TEAL
    elif metric == 'User hypothesis (stable USD)':
        val_color = GREY_TEXT

    ax5.text(COL_X[1], y, value, transform=ax5.transAxes,
             fontsize=8.5, fontweight='bold', color=val_color,
             fontfamily='Lora', va='center')

    ax5.text(COL_X[2], y, interp, transform=ax5.transAxes,
             fontsize=8, color=GREY_TEXT, fontfamily='Lora',
             va='center', style='italic')

# Thin rule under header
ax5.add_patch(mpl.patches.FancyBboxPatch(
    (0.0, HEADER_Y - 0.055), 1.0, 0.004,
    boxstyle='square,pad=0', linewidth=0,
    facecolor=RED, transform=ax5.transAxes, zorder=5, clip_on=False
))

# Bottom rule
bottom_rule_y = HEADER_Y - ROW_H * (len(table_data) + 1) + 0.02
ax5.add_patch(mpl.patches.FancyBboxPatch(
    (0.0, bottom_rule_y), 1.0, 0.004,
    boxstyle='square,pad=0', linewidth=0,
    facecolor=GREY_RULE, transform=ax5.transAxes, zorder=5, clip_on=False
))

add_economist_chrome(
    fig5, ax5,
    title    = "The numbers behind the peso's borrowed calm",
    subtitle = f'Key metrics, Philippine peso vs US dollar --- as of {today_str}',
    source   = 'Yahoo Finance (USDPHP=X, DX-Y.NYB); BSP; ARIMA(1,1,1) model'
)

fig5.savefig(f'{OUT_DIR}/chart_5_summary_table.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close(fig5)
print('  -> chart_5_summary_table.png')

print('\nAll 5 charts rendered.')
print(f'Directory: {OUT_DIR}')
