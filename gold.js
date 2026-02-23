    const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;
    const formatNumber = (value, digits = 2) => `${value.toFixed(digits)}`;
    const formatBp = (value) => `${value.toFixed(0)} bp`;
    const formatTonnes = (value) => `${value.toFixed(1)} t`;
    const formatCorr = (value) => `${value.toFixed(2)}`;
    const formatTimestamp = (isoString) => {
      const date = new Date(isoString);
      const month = date.toLocaleString('en-US', { month: 'short' });
      const day = date.getDate();
      const year = date.getFullYear();
      const time = date.toLocaleString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
      return `${month} ${day}, ${year} ${time}`;
    };

    const statusEl = document.getElementById('status');
    const statusSummaryEl = document.getElementById('statusSummary');
    const macroFollowThroughEl = document.getElementById('macroFollowThrough');
    const timestampsEl = document.getElementById('timestamps');
    const asOfEl = document.getElementById('as_of');
    const dataDatesEl = document.getElementById('data_dates');
    const confidenceEl = document.getElementById('confidence');
    const reasonLeadEl = document.getElementById('reasonLead');
    const reasonsEl = document.getElementById('reasons');
    const glanceCardsEl = document.getElementById('glanceCards');
    const escalationEl = document.getElementById('escalation');
    const horizonSummaryEl = document.getElementById('horizonSummary');
    const horizonRows = document.getElementById('horizonRows');
    const percentileNotesEl = document.getElementById('percentileNotes');
    const extensionRows = document.getElementById('extensionRows');
    const flowRows = document.getElementById('flowRows');
    const macroRows = document.getElementById('macroRows');
    const dailyStripEl = document.getElementById('dailyStrip');
    const portfolioLensEl = document.getElementById('portfolioLens');
    const lensTextEl = document.getElementById('lensText');
    const extensionSparklineEl = document.getElementById('extensionSparkline');
    const extensionSparklinePointsEl = document.getElementById('extensionSparklinePoints');
    const flowSparklineEl = document.getElementById('flowSparkline');
    const flowSparklinePointsEl = document.getElementById('flowSparklinePoints');
    const macroSparklineEl = document.getElementById('macroSparkline');
    const macroSparklinePointsEl = document.getElementById('macroSparklinePoints');
    const repoLink = document.getElementById('repoLink');
    const issueLink = document.getElementById('issueLink');
    const setExpandGroup = (group, expand) => {
      document.querySelectorAll(`[data-expand-group="${group}"]`).forEach((el) => {
        el.open = expand;
      });
    };
    document.querySelectorAll('[data-toggle-scope]').forEach((button) => {
      button.addEventListener('click', () => {
        const group = button.dataset.toggleScope;
        const action = button.dataset.toggleAction;
        setExpandGroup(group, action === 'expand');
      });
    });


    const contextTag = (text) => (text ? `<span class="context">${text}</span>` : '');
    const badgeDescriptions = {
      GREEN: 'Typical conditions.',
      BLUE: 'Extended rally; no confirmed deterioration.',
      ORANGE: 'Extended + early deterioration signals.',
      RED: 'Breakdown-style stress signals present.'
    };
    const CUT_LABELS = {
      CREDIBILITY_CUT: 'Credibility cut',
      STIMULUS_CUT: 'Stimulus cut',
      MIXED_CUT: 'Mixed cut',
      NO_CUTS_PRICED: 'No cuts priced',
      UNKNOWN: 'Unknown'
    };
    const cutWhyText = (label) => {
      switch (label) {
        case 'CREDIBILITY_CUT':
          return 'Cuts priced, but inflation expectations are falling faster than nominal yields → real yields tend to rise.';
        case 'STIMULUS_CUT':
          return 'Cuts priced with reflation impulse → breakevens rising and/or real yields falling.';
        case 'MIXED_CUT':
          return 'Cuts priced, but nominal and breakeven moves disagree (or are too small).';
        case 'NO_CUTS_PRICED':
          return 'Front-end rates are not falling enough to indicate meaningful cuts priced.';
        default:
          return 'Not enough data to classify.';
      }
    };
    const REASON_LABELS = {
      extension_ret_pctile_90: '3M return is extreme (≥90th percentile)',
      extension_200dma_pctile_90: '% above 200DMA is extreme (≥90th percentile)',
      extension_drawdown_pctile_70: '3M drawdown has been unusually shallow (≥70th percentile)',
      deterioration_flow_divergence: 'Holdings falling while price rising (flow divergence)',
      deterioration_macro_turn: 'Real yields rising quickly (macro headwind)',
      deterioration_policy_conflict: 'Cuts priced + real yields rising (credibility headwind)',
      deterioration_price_crack: 'Short-term weakness / drawdown worsening (price crack)',
      deterioration_followthrough: 'Macro follow-through (rates driving price)',
      composite_followthrough: 'Macro follow-through (rates driving price)'
    };
    const LENS_COPY = {
      conservative: 'Conservative lens: prioritize stability. A downgrade in regime matters more than upside extension.',
      balanced: 'Balanced lens: watch whether extension cools before deterioration broadens.',
      hardAsset: 'Hard-asset-heavy lens: focus on liquidation vs inflation signals before changing conviction.'
    };
    const STATE_RANK = { GREEN: 0, BLUE: 1, ORANGE: 2, RED: 3 };
    const HISTORY_KEY = 'gold-risk-monitor-history-v1';
    const percentileContext = (value) => {
      if (value == null) return 'N/A';
      if (value <= 20) return 'low';
      if (value <= 79) return 'typical';
      if (value <= 94) return 'high';
      return 'extreme';
    };
    const percentileWord = (value) => {
      const context = percentileContext(value);
      if (context === 'N/A') return 'N/A';
      return `${context[0].toUpperCase()}${context.slice(1)}`;
    };
    const dedupeNote = (note) => {
      if (!note) return note;
      return note.replace(/(\bused available history\b)(\s+\1)+/gi, '$1');
    };
    const escapeHtml = (value) => String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
    const formatOrdinal = (value) => {
      const rounded = Math.round(value);
      const mod10 = rounded % 10;
      const mod100 = rounded % 100;
      if (mod10 === 1 && mod100 !== 11) return `${rounded}st`;
      if (mod10 === 2 && mod100 !== 12) return `${rounded}nd`;
      if (mod10 === 3 && mod100 !== 13) return `${rounded}rd`;
      return `${rounded}th`;
    };
    const percentileText = (value, explain) => {
      if (value == null) {
        return `N/A${contextTag(explain || 'insufficient history')}`;
      }
      const cleaned = dedupeNote(explain);
      return `${percentileWord(value)} (${formatOrdinal(value)})${contextTag(cleaned ? `Note: ${cleaned}` : '')}`;
    };
    const valueOrNA = (value, formatter, explain) => {
      if (value == null) {
        return `N/A${contextTag(explain || 'insufficient history')}`;
      }
      return `${formatter(value)}${contextTag(explain)}`;
    };
    const signedValue = (value, formatter) => {
      if (value == null) return 'N/A';
      const sign = value > 0 ? '+' : '';
      return `${sign}${formatter(value)}`;
    };
    const formatDistance = (value, unit) => {
      if (value == null) return '—';
      if (unit === 'percent') return signedValue(value, (val) => `${(val * 100).toFixed(2)}%`);
      if (unit === 'bp') return signedValue(value, (val) => `${val.toFixed(0)} bp`);
      if (unit === 'pctile') return signedValue(value, (val) => `${Math.round(val)} pctile`);
      return signedValue(value, (val) => `${val}`);
    };
    const formatCurrent = (value, unit) => {
      if (value == null) return 'N/A';
      if (unit === 'percent') return formatPercent(value);
      if (unit === 'bp') return formatBp(value);
      if (unit === 'pctile') return `${formatOrdinal(value)} percentile`;
      return `${value}`;
    };
    const distanceDirectionText = (distance, unit) => {
      if (distance == null) return 'Need more data';
      if (distance === 0) return 'Trigger met';
      const move = distance > 0 ? 'rise' : 'fall';
      return `Needs ${move} by ${formatDistance(Math.abs(distance), unit).replace('+', '')}`;
    };
    const progressToTrigger = (row) => {
      if (row.fired) return 100;
      if (row.current == null || row.distance == null) return 0;
      const threshold = row.current + row.distance;
      const denominator = Math.abs(threshold) + 1e-9;
      const pct = 100 - (Math.abs(row.distance) / denominator) * 100;
      return Math.max(0, Math.min(99, pct));
    };
    const loadHistory = () => {
      try {
        const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        return Array.isArray(parsed) ? parsed : [];
      } catch (error) {
        return [];
      }
    };
    const saveHistory = (history) => {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-120)));
    };
    const pushSnapshot = (snapshot) => {
      const history = loadHistory();
      const next = history.filter((item) => item.updated_at !== snapshot.updated_at);
      next.push(snapshot);
      next.sort((a, b) => (a.updated_at < b.updated_at ? -1 : 1));
      saveHistory(next);
      return next;
    };
    const lastFivePoints = (series, formatter) => series.slice(-5).map((point) => `${point.date.slice(5)} ${formatter(point.value)}`).join(' • ');
    const formatDateTick = (isoDate) => {
      const date = new Date(isoDate);
      if (Number.isNaN(date.getTime())) {
        const fallback = new Date(`${isoDate}T00:00:00`);
        if (Number.isNaN(fallback.getTime())) return isoDate;
        return fallback.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };
    const formatDateTickWithYear = (isoDate) => {
      const date = new Date(isoDate);
      if (Number.isNaN(date.getTime())) {
        const fallback = new Date(`${isoDate}T00:00:00`);
        if (Number.isNaN(fallback.getTime())) return isoDate;
        return fallback.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      }
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    };
    const renderSparkline = (svgEl, pointsEl, series, formatter, axisLabels) => {
      if (!series.length) {
        svgEl.innerHTML = '';
        pointsEl.textContent = 'History builds as daily snapshots are saved.';
        return;
      }
      const values = series.map((item) => item.value);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || 1;
      const height = 96;
      const width = Math.max(240, Math.round(svgEl.clientWidth || 240));
      svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);
      const padding = { top: 10, right: 10, bottom: 34, left: 52 };
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const path = series.map((point, idx) => {
        const x = padding.left + (idx / Math.max(1, series.length - 1)) * plotWidth;
        const y = padding.top + (1 - ((point.value - min) / span)) * plotHeight;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');
      const xLabel = axisLabels?.x || 'Time';
      const yLabel = axisLabels?.y || 'Value';
      const xAxisY = height - padding.bottom + 0.5;
      const yAxisX = padding.left + 0.5;
      const xTickTarget = Math.min(5, series.length);
      const xTickIndices = Array.from({ length: xTickTarget }, (_, i) => {
        if (xTickTarget === 1) return series.length - 1;
        return Math.round((i / (xTickTarget - 1)) * (series.length - 1));
      }).filter((index, position, arr) => arr.indexOf(index) === position);
      const xTickCandidates = xTickIndices.map((index) => ({
        index,
        shortLabel: formatDateTick(series[index].date),
      }));
      const duplicateShortLabels = new Set(
        xTickCandidates
          .map((tick) => tick.shortLabel)
          .filter((label, position, arr) => arr.indexOf(label) !== position)
      );
      const xTicks = xTickCandidates.map((tick) => {
        const x = padding.left + (tick.index / Math.max(1, series.length - 1)) * plotWidth;
        const label = duplicateShortLabels.has(tick.shortLabel)
          ? formatDateTickWithYear(series[tick.index].date)
          : tick.shortLabel;
        return `
          <line class="sparkline-tick" x1="${x.toFixed(1)}" y1="${xAxisY}" x2="${x.toFixed(1)}" y2="${(xAxisY - 4).toFixed(1)}"></line>
          <text class="sparkline-axis-label" x="${x.toFixed(1)}" y="${(height - 14).toFixed(1)}" text-anchor="middle">${label}</text>
        `;
      }).join('');
      const yTickTarget = 4;
      const yTicks = Array.from({ length: yTickTarget }, (_, i) => {
        const ratio = i / Math.max(1, yTickTarget - 1);
        const y = padding.top + ratio * plotHeight;
        const value = max - ratio * span;
        return `
          <line class="sparkline-tick" x1="${yAxisX}" y1="${y.toFixed(1)}" x2="${(yAxisX + 4).toFixed(1)}" y2="${y.toFixed(1)}"></line>
          <text class="sparkline-axis-label" x="${(yAxisX - 6).toFixed(1)}" y="${(y + 3).toFixed(1)}" text-anchor="end">${formatter(value)}</text>
        `;
      }).join('');
      const axes = `
        <g class="sparkline-axes">
          <line class="sparkline-axis" x1="${padding.left}" y1="${xAxisY}" x2="${width - padding.right}" y2="${xAxisY}"></line>
          <line class="sparkline-axis" x1="${yAxisX}" y1="${padding.top}" x2="${yAxisX}" y2="${height - padding.bottom}"></line>
          ${xTicks}
          ${yTicks}
          <text class="sparkline-axis-label" x="${padding.left + plotWidth / 2}" y="${height - 2}" text-anchor="middle">${xLabel}</text>
          <text class="sparkline-axis-label" x="13" y="${padding.top + plotHeight / 2}" text-anchor="middle" transform="rotate(-90 13 ${padding.top + plotHeight / 2})">${yLabel}</text>
        </g>
      `;
      svgEl.innerHTML = `${axes}<polyline class="sparkline-polyline" points="${path}"></polyline>`;
      const hoverText = `Last 5 points: ${lastFivePoints(series, formatter)}`;
      svgEl.onmouseenter = () => { pointsEl.textContent = hoverText; };
      svgEl.onmouseleave = () => { pointsEl.textContent = 'Hover to see last 5 points.'; };
    };
    const updateLensCopy = () => {
      lensTextEl.textContent = LENS_COPY[portfolioLensEl.value] || LENS_COPY.balanced;
    };
    portfolioLensEl.addEventListener('change', updateLensCopy);
    updateLensCopy();
    const buildTransitionTable = (title, rows, note) => {
      if (!rows || rows.length === 0) {
        return `
          <details class="expandable-card" data-expand-group="mechanics">
            <summary>${title}</summary>
            <div class="expandable-body meta">No transition data available.</div>
          </details>
        `;
      }
      const noteHtml = note ? `<div class="meta">${note}</div>` : '';
      const unmet = rows.filter((row) => row.fired === false && row.distance != null);
      const weakest = unmet.sort((a, b) => Math.abs(a.distance) - Math.abs(b.distance))[0];
      const weakestHtml = weakest
        ? `<div class="weakest-link"><strong>Closest to flipping:</strong> ${weakest.trigger} (${distanceDirectionText(weakest.distance, weakest.unit)})</div>`
        : '<div class="weakest-link">All listed triggers are currently met.</div>';
      const bodyRows = rows.map((row) => {
        const progress = progressToTrigger(row);
        return `
          <tr>
            <td>${row.trigger}${row.note ? contextTag(row.note) : ''}</td>
            <td class="num">${formatCurrent(row.current, row.unit)}</td>
            <td class="num">${row.threshold}</td>
            <td class="num">${formatDistance(row.distance, row.unit)}</td>
            <td>${distanceDirectionText(row.distance, row.unit)}</td>
            <td>
              <div class="meta">${progress.toFixed(0)}% to trigger</div>
              <div class="trigger-progress"><div class="trigger-progress-bar" style="width:${progress.toFixed(0)}%"></div></div>
            </td>
          </tr>
        `;
      }).join('');
      return `
        <details class="expandable-card" data-expand-group="mechanics">
          <summary>${title}</summary>
          <div class="expandable-body control-panel-grid">
            ${noteHtml}
            ${weakestHtml}
            <div class="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th class="num">Current</th>
                    <th class="num">Trigger</th>
                    <th class="num">Distance</th>
                    <th>Direction needed</th>
                    <th>Progress</th>
                  </tr>
                </thead>
                <tbody>
                  ${bodyRows}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      `;
    };
    const buildCard = (title, value, percentile, label, why) => `
      <div class="metric-card">
        <h3>${title}</h3>
        <div class="value">${value}</div>
        <div class="subtle">${label}${percentile != null ? ` • ${percentileWord(percentile)} (${formatOrdinal(percentile)})` : ''}</div>
        <div class="why">${why}</div>
      </div>
    `;
    fetch('data.json', { cache: 'no-store' })
      .then((response) => response.json())
      .then((data) => {
        const regime = data.regime || {};
        const state = regime.state || data.flag || 'UNKNOWN';
        statusEl.textContent = state;
        statusEl.classList.remove('green', 'blue', 'orange', 'red');
        statusEl.classList.add(state.toLowerCase());
        statusSummaryEl.textContent = badgeDescriptions[state] || 'Status summary unavailable.';
        const macroFollowThrough = regime.flags?.macro_followthrough;
        macroFollowThroughEl.textContent = macroFollowThrough
          ? 'Macro shock is sticking (rates are driving price).'
          : 'Macro shock not confirmed (moves look more liquidation/positioning).';

        if (data.updated_at) {
          const etFormatted = formatTimestamp(data.updated_at);
          timestampsEl.textContent = `Updated ${etFormatted}`;
          asOfEl.textContent = etFormatted;
        }

        const sourceCount = Object.keys(data.sources || {}).length;
        dataDatesEl.textContent = sourceCount
          ? `${sourceCount} inputs refreshed`
          : 'Daily close inputs';
        confidenceEl.textContent = state === 'RED' ? 'Elevated risk' : 'Medium';

        const reasons = regime.reasons || data.flag_triggers || [];
        reasonLeadEl.innerHTML = `<strong>Why ${state} today</strong>`;
        reasonsEl.innerHTML = reasons.length
          ? reasons.map((item) => {
            const label = escapeHtml(REASON_LABELS[item] || item);
            const reasonCode = escapeHtml(item);
            return `
              <li>
                <span class="reason-item">
                  <span>${label}</span>
                  <span class="reason-help" tabindex="0" aria-label="Show trigger code for ${label}">ⓘ
                    <span class="reason-help-tooltip">Trigger code: <code>${reasonCode}</code></span>
                  </span>
                </span>
              </li>
            `;
          }).join('')
          : '<li>No active triggers.</li>';

        const metrics = data.metrics || {};
        const horizons = metrics.horizons || {};
        const horizonOrder = ['3M', '6M', '1Y', '3Y', '5Y'];
        horizonRows.innerHTML = horizonOrder.map((label) => {
          const horizon = horizons[label] || {};
          const available = horizon.available !== false && horizon.ret != null;
          const returnCell = available
            ? formatPercent(horizon.ret)
            : `N/A${contextTag(horizon.ret_pctile_5y_explain || 'insufficient history')}`;
          const returnPctCell = available
            ? percentileText(
              horizon.ret_pctile_5y,
              horizon.ret_pctile_5y_explain
            )
            : `N/A${contextTag(horizon.ret_pctile_5y_explain || 'insufficient history')}`;
          const drawdownCell = available
            ? formatPercent(horizon.max_drawdown)
            : `N/A${contextTag(horizon.max_drawdown_pctile_5y_explain || 'insufficient history')}`;
          const drawdownPctCell = available
            ? percentileText(
              horizon.max_drawdown_pctile_5y,
              horizon.max_drawdown_pctile_5y_explain
            )
            : `N/A${contextTag(horizon.max_drawdown_pctile_5y_explain || 'insufficient history')}`;
          const contextValue = percentileContext(horizon.ret_pctile_5y ?? horizon.max_drawdown_pctile_5y);
          return `
            <tr>
              <td>${label}</td>
              <td class="num">${returnCell}</td>
              <td class="num">${returnPctCell}</td>
              <td class="num">${drawdownCell}</td>
              <td class="num">${drawdownPctCell}</td>
              <td>${contextValue}</td>
            </tr>
          `;
        }).join('');

        const percentileNotes = data.notes?.percentile_notes || {};
        const notesEntries = Object.entries(percentileNotes);
        const notesText = notesEntries.length
          ? notesEntries.map(([key, value]) => `${key}: ${dedupeNote(value)}`).join(' | ')
          : 'none';
        const needsDetails = notesText.length > 140 || notesEntries.length > 2;
        percentileNotesEl.innerHTML = needsDetails
          ? `<details><summary>Percentile notes</summary>${notesText}</details>`
          : `<strong>Percentile notes:</strong> ${notesText}`;

        extensionRows.innerHTML = `
          <tr>
            <td>GLD % above 200DMA</td>
            <td class="num">${valueOrNA(metrics.pct_above_200dma, formatPercent)}</td>
            <td class="num">${percentileText(
              metrics.pct_above_200dma_pctile_5y,
              metrics.pct_above_200dma_pctile_5y_explain
            )}</td>
          </tr>
        `;

        const flows = metrics.flows || {};
        flowRows.innerHTML = `
          <tr>
            <td>Holdings today (tonnes)</td>
            <td class="num">${valueOrNA(flows.holdings_today_tonnes, formatTonnes)}</td>
            <td class="num">N/A</td>
          </tr>
          <tr>
            <td>Holdings change (5D)</td>
            <td class="num">${valueOrNA(flows.holdings_change_5d_pct, formatPercent)}</td>
            <td class="num">${percentileText(
              flows.holdings_change_5d_pct_pctile_5y,
              flows.holdings_change_5d_pct_pctile_5y_explain
            )}</td>
          </tr>
          <tr>
            <td>Holdings change (21D)</td>
            <td class="num">${valueOrNA(flows.holdings_change_21d_pct, formatPercent)}</td>
            <td class="num">${percentileText(
              flows.holdings_change_21d_pct_pctile_5y,
              flows.holdings_change_21d_pct_pctile_5y_explain
            )}</td>
          </tr>
        `;

        const macro = metrics.macro || {};
        const cut = macro.cut_style_classifier || {};
        const cutInputs = cut.inputs || {};
        const cutLabel = CUT_LABELS[cut.label] || cut.label || 'Unknown';
        const cutExplain = cutWhyText(cut.label || 'UNKNOWN');
        const cutDetails = (cut.explain && cut.explain.length)
          ? `<details><summary>Classifier details</summary>${cut.explain.map((item) => `<div class="context">${item}</div>`).join('')}</details>`
          : '';

        const snapshot = {
          updated_at: data.updated_at || new Date().toISOString(),
          state,
          drivers: {
            extension_ret_pctile_90: horizons['3M']?.ret_pctile_5y,
            extension_200dma_pctile_90: metrics.pct_above_200dma_pctile_5y,
            extension_drawdown_pctile_70: horizons['3M']?.max_drawdown_pctile_5y,
            deterioration_flow_divergence: flows.holdings_change_21d_pct_pctile_5y,
            deterioration_macro_turn: macro.real_yield_change_1m_bp_pctile_5y,
            deterioration_followthrough: macro.corr_gld_ret_vs_real_yield_chg_20d_pctile_5y,
          },
          spark: {
            extension: horizons['3M']?.ret,
            flows: flows.holdings_change_21d_pct,
            macro: macro.real_yield_change_1m_bp,
          },
        };
        const history = pushSnapshot(snapshot);
        const previous = history.length > 1 ? history[history.length - 2] : null;
        const regimeDelta = (() => {
          if (!previous) return 'insufficient history';
          const prevRank = STATE_RANK[previous.state] ?? 0;
          const currentRank = STATE_RANK[state] ?? 0;
          if (currentRank === prevRank) return 'unchanged';
          return currentRank > prevRank ? 'upgraded' : 'downgraded';
        })();
        const driverMovers = Object.entries(snapshot.drivers)
          .map(([key, value]) => ({
            key,
            label: REASON_LABELS[key] || key,
            now: value,
            prev: previous?.drivers?.[key],
            delta: previous?.drivers?.[key] == null || value == null ? null : value - previous.drivers[key],
          }))
          .filter((item) => item.delta != null)
          .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
          .slice(0, 3);
        const moverText = driverMovers.length
          ? driverMovers.map((item) => `${item.label} (${item.delta > 0 ? '+' : ''}${Math.round(item.delta)} pctile)`).join(' • ')
          : 'Top movers will appear after one additional daily snapshot.';
        const whyText = driverMovers.length
          ? `Regime ${regimeDelta} as ${driverMovers[0].label.toLowerCase()} moved ${driverMovers[0].delta > 0 ? 'higher' : 'lower'} vs yesterday.`
          : 'Why-changed sentence will populate once yesterday data is available in local history.';
        dailyStripEl.innerHTML = `
          <div class="daily-chip"><span class="daily-chip-title">Regime vs yesterday</span><strong>${regimeDelta}</strong></div>
          <div class="daily-chip"><span class="daily-chip-title">Top 3 percentile movers</span>${moverText}</div>
          <div class="daily-chip"><span class="daily-chip-title">Why changed</span>${whyText}</div>
        `;

        const sparkHistory = history.slice(-90);
        renderSparkline(
          extensionSparklineEl,
          extensionSparklinePointsEl,
          sparkHistory.filter((item) => item.spark.extension != null).map((item) => ({ date: item.updated_at, value: item.spark.extension })),
          formatPercent,
          { x: 'Date', y: 'Return %' }
        );
        renderSparkline(
          flowSparklineEl,
          flowSparklinePointsEl,
          sparkHistory.filter((item) => item.spark.flows != null).map((item) => ({ date: item.updated_at, value: item.spark.flows })),
          formatPercent,
          { x: 'Date', y: 'Flow %' }
        );
        renderSparkline(
          macroSparklineEl,
          macroSparklinePointsEl,
          sparkHistory.filter((item) => item.spark.macro != null).map((item) => ({ date: item.updated_at, value: item.spark.macro })),
          formatBp,
          { x: 'Date', y: 'Real yld bp' }
        );
        macroRows.innerHTML = `
          <tr>
            <td>Real yield today (DFII10)</td>
            <td class="num">${valueOrNA(macro.real_yield_today, (value) => `${value.toFixed(2)}%`)}</td>
            <td class="num">N/A</td>
          </tr>
          <tr>
            <td>Real yield change (1M)</td>
            <td class="num">${valueOrNA(macro.real_yield_change_1m_bp, formatBp)}</td>
            <td class="num">${percentileText(
              macro.real_yield_change_1m_bp_pctile_5y,
              macro.real_yield_change_1m_bp_pctile_5y_explain
            )}</td>
          </tr>
          <tr>
            <td>Real yield change (3M)</td>
            <td class="num">${valueOrNA(macro.real_yield_change_3m_bp, formatBp)}</td>
            <td class="num">${percentileText(
              macro.real_yield_change_3m_bp_pctile_5y,
              macro.real_yield_change_3m_bp_pctile_5y_explain
            )}</td>
          </tr>
          <tr>
            <td>20D corr (GLD ret vs real-yield Δ)</td>
            <td class="num">${valueOrNA(macro.corr_gld_ret_vs_real_yield_chg_20d, formatCorr)}</td>
            <td class="num">${percentileText(
              macro.corr_gld_ret_vs_real_yield_chg_20d_pctile_5y,
              macro.corr_gld_ret_vs_real_yield_chg_20d_pctile_5y_explain
            )}</td>
          </tr>

          <tr>
            <td colspan="3"><div class="divider"></div></td>
          </tr>
          <tr>
            <td>Cut style classifier</td>
            <td class="num">${cutLabel}${contextTag(cutExplain)}${cutDetails}</td>
            <td class="num">N/A</td>
          </tr>
          <tr>
            <td>Nominal 2Y change (1M)</td>
            <td class="num">${valueOrNA(cutInputs.nominal_2y_change_1m_bp, formatBp)}</td>
            <td class="num">N/A</td>
          </tr>
          <tr>
            <td>Nominal 10Y change (1M)</td>
            <td class="num">${valueOrNA(cutInputs.nominal_10y_change_1m_bp, formatBp)}</td>
            <td class="num">N/A</td>
          </tr>
          <tr>
            <td>Breakeven 10Y change (1M)</td>
            <td class="num">${valueOrNA(cutInputs.breakeven_10y_change_1m_bp, formatBp)}</td>
            <td class="num">N/A</td>
          </tr>
          <tr>
            <td>Disinflation ratio (|ΔBE|/|ΔNom10|)</td>
            <td class="num">${cutInputs.disinflation_ratio == null ? 'N/A' : formatNumber(cutInputs.disinflation_ratio, 2)}</td>
            <td class="num">N/A</td>
          </tr>
        `;
        const glanceCards = [
          buildCard(
            'Extension',
            valueOrNA(metrics.pct_above_200dma, formatPercent),
            metrics.pct_above_200dma_pctile_5y,
            'vs 200DMA',
            'How stretched price is vs trend'
          ),
          buildCard(
            '3M Return',
            valueOrNA(horizons['3M']?.ret, formatPercent),
            horizons['3M']?.ret_pctile_5y,
            'total return',
            'Recent momentum'
          ),
          buildCard(
            'Flows (21D)',
            valueOrNA(flows.holdings_change_21d_pct, formatPercent),
            flows.holdings_change_21d_pct_pctile_5y,
            '21D change',
            'ETF demand proxy'
          ),
          buildCard(
            'Real yields (1M)',
            valueOrNA(macro.real_yield_change_1m_bp, formatBp),
            macro.real_yield_change_1m_bp_pctile_5y,
            '1M change',
            'Macro pressure proxy'
          ),
          buildCard(
            'Macro link',
            valueOrNA(macro.corr_gld_ret_vs_real_yield_chg_20d, formatCorr),
            macro.corr_gld_ret_vs_real_yield_chg_20d_pctile_5y,
            '20D corr',
            'Is selling macro-driven? (more negative = more rate-linked)'
          ),
          (() => {
            const label = cut.label || 'UNKNOWN';
            const pretty = CUT_LABELS[label] || label;
            const ratio = cutInputs.disinflation_ratio;
            const ratioText = ratio == null ? 'N/A' : `${ratio.toFixed(2)}×`;
            return buildCard(
              'Cut style',
              pretty,
              null,
              ratio == null ? 'classifier' : `BE/Nom ${ratioText}`,
              cutWhyText(label)
            );
          })()
        ];
        glanceCardsEl.innerHTML = glanceCards.join('');

        const transitions = regime.transitions || {};
        const deescalateRows = transitions.deescalate_to_blue || [];
        const redPrimaryRows = transitions.escalate_to_red_primary || [];
        const redCompositeRows = transitions.escalate_to_red_composite || [];
        const normalizeRows = transitions.normalize_to_green || [];

        const buildLocalRow = ({ trigger, thresholdLabel, current, thresholdValue, fired, unit, note }) => {
          let distance = null;
          if (current != null && thresholdValue != null) {
            distance = fired ? 0 : thresholdValue - current;
          }
          return {
            trigger,
            threshold: thresholdLabel,
            current,
            fired,
            distance,
            unit,
            note,
          };
        };

        const primaryRetPctile = horizons['3M']?.ret_pctile_5y;
        const drawdownPctile = horizons['3M']?.max_drawdown_pctile_5y;
        const extensionSignalRows = [
          buildLocalRow({
            trigger: 'Extension: 3M return percentile',
            thresholdLabel: '≥ 90th percentile',
            current: primaryRetPctile,
            thresholdValue: 90,
            fired: primaryRetPctile != null && primaryRetPctile >= 90,
            unit: 'pctile',
          }),
          buildLocalRow({
            trigger: 'Extension: % above 200DMA percentile',
            thresholdLabel: '≥ 90th percentile',
            current: metrics.pct_above_200dma_pctile_5y,
            thresholdValue: 90,
            fired: metrics.pct_above_200dma_pctile_5y != null && metrics.pct_above_200dma_pctile_5y >= 90,
            unit: 'pctile',
          }),
          buildLocalRow({
            trigger: 'Extension: 3M drawdown percentile',
            thresholdLabel: '≥ 70th percentile',
            current: drawdownPctile,
            thresholdValue: 70,
            fired: drawdownPctile != null && drawdownPctile >= 70,
            unit: 'pctile',
          }),
        ];

        const deteriorationTitle = state === 'GREEN'
          ? 'Hold position: deterioration signals (required for ORANGE once BLUE is active)'
          : 'Move down: de-escalate to BLUE (no deterioration triggers)';
        const deteriorationNote = state === 'GREEN'
          ? 'ORANGE requires BLUE plus at least one deterioration signal. If extension signals are not met, these do not change the color.'
          : 'All deterioration signals must clear for the regime to return to BLUE.';

        const extensionTitle = state === 'GREEN'
          ? 'Move up: escalate to BLUE (extension signals)'
          : 'Hold position: extension signals (still required for BLUE)';
        const extensionNote = 'BLUE requires at least 2 of the 3 extension signals.';

        escalationEl.innerHTML = `
          ${buildTransitionTable(
            extensionTitle,
            extensionSignalRows,
            extensionNote
          )}
          ${buildTransitionTable(
            deteriorationTitle,
            deescalateRows,
            deteriorationNote
          )}
          ${buildTransitionTable(
            'Move up: escalate to RED (primary triggers)',
            redPrimaryRows,
            'Any single primary trigger can move the regime to RED.'
          )}
          ${buildTransitionTable(
            'Move up: escalate to RED (composite: 2 of 5)',
            redCompositeRows,
            'Two or more composite signals would also trigger RED.'
          )}
          ${buildTransitionTable(
            'Move down: normalize to GREEN',
            normalizeRows,
            'Normalization requires percentile conditions to reset across price, flows, and macro.'
          )}
        `;

        const shortTermContext = percentileContext(horizons['3M']?.ret_pctile_5y);
        const longTermContext = percentileContext(horizons['1Y']?.ret_pctile_5y);
        horizonSummaryEl.textContent = `Context only: short-term ${shortTermContext}; long-term ${longTermContext}.`;

        const repo = data.repository || 'https://github.com/';
        repoLink.href = repo;

        if (data.latest_issue_url) {
          issueLink.innerHTML = ` Latest issue: <a href="${data.latest_issue_url}">${data.latest_issue_url}</a>`;
        }
      })
      .catch((error) => {
        statusEl.textContent = 'DATA ERROR';
        statusEl.classList.add('red');
        horizonRows.innerHTML = `<tr><td colspan="6">Failed to load data.json: ${error}</td></tr>`;
      });
