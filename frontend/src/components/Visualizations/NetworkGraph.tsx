import { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import type { NetworkGraph as GraphData, NetworkNode } from '../../types';

const LAYER_TYPE_COLORS: Record<string, string> = {
  input: '#3b82f6',
  hidden: '#10b981',
  output: '#ef4444',
  conv: '#8b5cf6',
  fc: '#6366f1',
  rnn: '#f59e0b',
  lstm: '#ec4899',
  generator: '#14b8a6',
  discriminator: '#f97316',
  gen_output: '#06b6d4',
  latent: '#a78bfa',
  attention: '#fb923c',
  embedding: '#818cf8',
  feedforward: '#34d399',
  encoder: '#2dd4bf',
  decoder: '#fb7185',
  bottleneck: '#c084fc',
};

const EDIT_COLOR = '#f59e0b';
const SELECT_COLOR = '#fde047';

/**
 * - architecture / forward / backward: the original Neural Visualizer modes.
 * - signal: edges are drawn by their real contribution w·a_source on the
 *   current probe (requires node values to come from a real model), nodes by
 *   |activation| relative to their layer.  Used by the Neural Microscope.
 */
export type GraphMode = 'architecture' | 'forward' | 'backward' | 'signal';

interface Props {
  graph: GraphData;
  activeNodeIds?: Set<number>;
  activeEdgeIds?: Set<number>;
  gradients?: Record<string, number>;
  mode?: GraphMode;
  // ── Microscope extensions (all optional) ──
  onNodeClick?: (node: NetworkNode) => void;
  onLayerClick?: (layer: number) => void;
  selectedNodeIds?: Set<number>;
  selectedEdgeIds?: Set<number>;
  selectedLayer?: number | null;
  layerLabels?: Record<number, { title: string; subtitle?: string }>;
  showHint?: boolean;
}

interface DrawState {
  g: d3.Selection<SVGGElement, unknown, null, undefined>;
  x: d3.ScaleLinear<number, number>;
  y: d3.ScaleLinear<number, number>;
}

export function NetworkGraph({
  graph, activeNodeIds, activeEdgeIds, gradients, mode = 'architecture',
  onNodeClick, onLayerClick, selectedNodeIds, selectedEdgeIds, selectedLayer, layerLabels, showHint = true,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const drawRef = useRef<DrawState | null>(null);
  // Click handlers live in refs so a new callback identity never forces a full redraw.
  const nodeClickRef = useRef(onNodeClick);
  const layerClickRef = useRef(onLayerClick);
  useEffect(() => {
    nodeClickRef.current = onNodeClick;
    layerClickRef.current = onLayerClick;
  });

  // Compute scale to fit all nodes
  const { minX, maxX, minY, maxY } = useMemo(() => {
    if (!graph.nodes.length) return { minX: 0, maxX: 10, minY: -5, maxY: 5 };
    const xs = graph.nodes.map((n) => n.x);
    const ys = graph.nodes.map((n) => n.y);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }, [graph.nodes]);

  const interactive = !!onNodeClick;
  const layerInteractive = !!onLayerClick;

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !graph.nodes.length) return;

    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;
    const pad = 60;
    const topPad = layerLabels ? 78 : pad;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    // Defs for glow filters and arrowheads
    const defs = svg.append('defs');

    const glowFilter = defs.append('filter').attr('id', 'glow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    glowFilter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    const activeGlow = defs.append('filter').attr('id', 'activeGlow').attr('x', '-100%').attr('y', '-100%').attr('width', '300%').attr('height', '300%');
    activeGlow.append('feGaussianBlur').attr('stdDeviation', '5').attr('result', 'coloredBlur');
    const feMerge2 = activeGlow.append('feMerge');
    feMerge2.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge2.append('feMergeNode').attr('in', 'SourceGraphic');

    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 10)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#4b5563');

    // Scales
    const xScale = d3.scaleLinear().domain([minX, maxX]).range([pad, width - pad]);
    const yScale = d3.scaleLinear().domain([minY, maxY]).range([height - pad, topPad]);

    const nodeById = new Map<number, NetworkNode>(graph.nodes.map((n) => [n.id, n]));

    // Signal mode: normalise per layer so every layer is readable.
    const maxContribByLayer = new Map<number, number>();
    const maxValueByLayer = new Map<number, number>();
    if (mode === 'signal') {
      graph.edges.forEach((e) => {
        const c = Math.abs(e.weight * (nodeById.get(e.source)?.value ?? 0));
        maxContribByLayer.set(e.layer, Math.max(maxContribByLayer.get(e.layer) ?? 0, c));
      });
      graph.nodes.forEach((n) => {
        maxValueByLayer.set(n.layer, Math.max(maxValueByLayer.get(n.layer) ?? 0, Math.abs(n.value ?? 0)));
      });
    }

    // Background grid
    const gridG = svg.append('g').attr('class', 'grid').attr('opacity', 0.15);
    const gridSpacing = 40;
    for (let x = 0; x < width; x += gridSpacing) {
      gridG.append('line').attr('x1', x).attr('y1', 0).attr('x2', x).attr('y2', height).attr('stroke', '#374151').attr('stroke-width', 0.5);
    }
    for (let y = 0; y < height; y += gridSpacing) {
      gridG.append('line').attr('x1', 0).attr('y1', y).attr('x2', width).attr('y2', y).attr('stroke', '#374151').attr('stroke-width', 0.5);
    }

    const g = svg.append('g');

    // Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);

    // Layer column headers (clickable in the microscope)
    if (layerLabels) {
      const layerX = new Map<number, number>();
      graph.nodes.forEach((n) => layerX.set(n.layer, n.x));
      const headerG = g.append('g').attr('class', 'layer-headers');
      layerX.forEach((lx, layer) => {
        const label = layerLabels[layer];
        if (!label) return;
        const hg = headerG.append('g')
          .attr('transform', `translate(${xScale(lx)}, ${topPad - 50})`)
          .attr('cursor', layerInteractive ? 'pointer' : 'default')
          .attr('data-layer', layer);
        hg.append('rect')
          .attr('x', -46).attr('y', -12).attr('width', 92).attr('height', 32).attr('rx', 6)
          .attr('fill', 'rgba(17,24,39,0.85)').attr('stroke', '#374151');
        hg.append('text').attr('text-anchor', 'middle').attr('y', 1)
          .attr('font-size', '10px').attr('font-weight', 600).attr('fill', '#e2e8f0').text(label.title);
        if (label.subtitle) {
          hg.append('text').attr('text-anchor', 'middle').attr('y', 13)
            .attr('font-size', '8.5px').attr('fill', '#9ca3af').text(label.subtitle);
        }
        hg.on('click', (event) => {
          event.stopPropagation();
          layerClickRef.current?.(layer);
        });
      });
    }

    // Edges
    const edgeG = g.append('g').attr('class', 'edges');
    graph.edges.forEach((edge, i) => {
      const src = nodeById.get(edge.source);
      const tgt = nodeById.get(edge.target);
      if (!src || !tgt) return;

      const isActive = activeEdgeIds ? activeEdgeIds.has(i) : true;
      const isSkip = edge.type === 'skip';
      const isRecurrent = edge.type === 'recurrent';
      const isAttention = edge.type === 'attention';
      const weight = edge.weight;

      // Gradient flow colouring in backward mode
      let edgeGradMag = 0;
      if (mode === 'backward' && gradients && isActive) {
        const gSrc = gradients[String(edge.source)] ?? 0;
        const gTgt = gradients[String(edge.target)] ?? 0;
        edgeGradMag = (Math.abs(gSrc) + Math.abs(gTgt)) / 2;
      }

      const gradFlowColor = (() => {
        if (mode !== 'backward' || !gradients || !isActive) return null;
        if (edgeGradMag < 0.01) return '#3b82f6';   // vanishing — blue
        if (edgeGradMag < 0.1)  return '#10b981';   // healthy  — green
        if (edgeGradMag < 0.5)  return '#f59e0b';   // large    — yellow
        return '#ef4444';                             // exploding — red
      })();

      if (mode === 'signal') {
        const contrib = weight * (src.value ?? 0);
        const t = Math.abs(contrib) / (maxContribByLayer.get(edge.layer) || 1);
        edgeG.append('line')
          .attr('x1', xScale(src.x)).attr('y1', yScale(src.y))
          .attr('x2', xScale(tgt.x)).attr('y2', yScale(tgt.y))
          .attr('stroke', edge.edited ? EDIT_COLOR : contrib >= 0 ? '#10b981' : '#ef4444')
          .attr('stroke-width', edge.edited ? 2 : 0.4 + t * 3.2)
          .attr('stroke-opacity', edge.edited ? 0.95 : 0.05 + t * 0.85)
          .attr('stroke-dasharray', edge.edited ? '5,3' : 'none');
        return;
      }

      const baseColor = weight > 0 ? '#10b981' : '#ef4444';
      const opacity = isActive ? (mode === 'architecture' ? Math.min(0.8, 0.2 + Math.abs(weight) * 0.6) : 0.9) : 0.08;
      const strokeWidth = mode === 'architecture'
        ? Math.max(0.5, Math.min(3, Math.abs(weight) * 2))
        : isActive ? (mode === 'backward' ? Math.max(1, Math.min(4, edgeGradMag * 20)) : 2.5) : 0.5;

      const strokeColor = edge.edited ? EDIT_COLOR : gradFlowColor
        ?? (isSkip ? '#3b82f6' : isRecurrent ? '#f59e0b' : isAttention ? '#fb923c' : isActive ? '#60a5fa' : baseColor);

      const line = edgeG.append('line')
        .attr('x1', xScale(src.x))
        .attr('y1', yScale(src.y))
        .attr('x2', xScale(tgt.x))
        .attr('y2', yScale(tgt.y))
        .attr('stroke', strokeColor)
        .attr('stroke-width', strokeWidth)
        .attr('stroke-opacity', opacity)
        .attr('stroke-dasharray', isSkip || isRecurrent || edge.edited ? '4,3' : isAttention ? '2,3' : 'none');

      if (isActive && mode !== 'architecture') {
        line.attr('filter', 'url(#glow)');
        // Animated signal dot
        const circle = g.append('circle').attr('r', 3).attr('fill', '#60a5fa').attr('opacity', 0);
        circle.append('animateMotion')
          .attr('dur', `${0.8 + (i % 5) * 0.1}s`)
          .attr('repeatCount', 'indefinite')
          .attr('path', `M${xScale(src.x)},${yScale(src.y)} L${xScale(tgt.x)},${yScale(tgt.y)}`);
        circle.append('animate')
          .attr('attributeName', 'opacity')
          .attr('values', '0;1;0')
          .attr('dur', `${0.8 + (i % 5) * 0.1}s`)
          .attr('repeatCount', 'indefinite');
      }
    });

    // Nodes
    const nodeG = g.append('g').attr('class', 'nodes');
    const tooltip = d3.select(container).append('div')
      .style('position', 'absolute')
      .style('background', 'rgba(17, 24, 39, 0.95)')
      .style('border', '1px solid #374151')
      .style('border-radius', '8px')
      .style('padding', '8px 12px')
      .style('font-size', '12px')
      .style('color', '#e2e8f0')
      .style('pointer-events', 'none')
      .style('opacity', '0')
      .style('transition', 'opacity 0.15s')
      .style('z-index', '10')
      .style('max-width', '220px');

    graph.nodes.forEach((node) => {
      const cx = xScale(node.x);
      const cy = yScale(node.y);
      const isActive = activeNodeIds ? activeNodeIds.has(node.id) : true;
      const color = node.ablated ? '#4b5563' : LAYER_TYPE_COLORS[node.layer_type] ?? '#6b7280';
      const radius = node.layer_type === 'output' ? 14 : node.layer_type === 'input' ? 12 : 10;
      const highlighted = isActive && mode !== 'architecture' && mode !== 'signal';

      const nodeGroup = nodeG.append('g')
        .attr('transform', `translate(${cx}, ${cy})`)
        .attr('cursor', 'pointer');

      // Outer glow ring for active nodes
      if (highlighted) {
        nodeGroup.append('circle')
          .attr('r', radius + 6)
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 2)
          .attr('opacity', 0.3)
          .attr('filter', 'url(#activeGlow)');
      }

      // Value-based fill opacity
      let fillOpacity = mode === 'architecture' ? 0.85 : isActive ? 1.0 : 0.2;
      if (mode === 'signal') {
        const m = maxValueByLayer.get(node.layer) || 1;
        fillOpacity = node.ablated ? 0.5 : 0.18 + 0.82 * Math.min(1, Math.abs(node.value ?? 0) / m);
      }

      // Main circle
      nodeGroup.append('circle')
        .attr('r', radius)
        .attr('fill', color)
        .attr('fill-opacity', fillOpacity)
        .attr('stroke', node.edited ? EDIT_COLOR : highlighted || mode === 'signal' ? '#ffffff' : '#374151')
        .attr('stroke-opacity', mode === 'signal' && !node.edited ? 0.35 : 1)
        .attr('stroke-width', node.edited ? 2.5 : highlighted ? 2 : 1)
        .attr('filter', highlighted ? 'url(#glow)' : 'none');

      if (node.ablated) {
        // A disabled neuron: red cross
        const r = radius * 0.65;
        nodeGroup.append('path')
          .attr('d', `M${-r},${-r}L${r},${r}M${r},${-r}L${-r},${r}`)
          .attr('stroke', '#ef4444').attr('stroke-width', 2.2).attr('stroke-linecap', 'round')
          .attr('pointer-events', 'none');
      } else if (node.value !== undefined && mode !== 'signal') {
        // Value bar inside node
        const barH = Math.min(1, Math.abs(node.value)) * (radius - 2);
        nodeGroup.append('rect')
          .attr('x', -3)
          .attr('y', node.value >= 0 ? -barH : 0)
          .attr('width', 6)
          .attr('height', barH)
          .attr('fill', 'white')
          .attr('fill-opacity', 0.4)
          .attr('rx', 2);
      }

      // Gradient label (backward mode)
      if (mode === 'backward' && gradients && gradients[node.id] !== undefined) {
        const grad = gradients[node.id];
        nodeGroup.append('text')
          .attr('y', radius + 14)
          .attr('text-anchor', 'middle')
          .attr('font-size', '9px')
          .attr('fill', '#fb923c')
          .text(`∇${grad.toFixed(3)}`);
      }

      // Node label
      nodeGroup.append('text')
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .attr('font-size', '8px')
        .attr('font-weight', '600')
        .attr('fill', 'white')
        .attr('pointer-events', 'none')
        .text(node.ablated ? '' : node.name.length > 8 ? node.name.slice(0, 6) + '…' : node.name);

      // Tooltip
      nodeGroup
        .on('mouseenter', () => {
          tooltip.style('opacity', '1');
          const lines = [
            `<strong style="color:#60a5fa">${node.name}</strong>`,
            `Type: <span style="color:#a3e635">${node.layer_type}</span>`,
            `Value: <span style="color:#34d399">${node.value?.toFixed(4)}</span>`,
            node.z_val !== undefined && node.z_val !== null ? `z: <span style="color:#93c5fd">${node.z_val.toFixed(4)}</span>` : null,
            node.activation ? `Activation: <span style="color:#fb923c">${node.activation}</span>` : null,
            node.bias !== undefined && node.bias !== null ? `Bias: <span style="color:#c084fc">${node.bias.toFixed(4)}</span>` : null,
            node.ablated ? `<span style="color:#f87171">Disabled (output forced to 0)</span>` : null,
            interactive ? `<span style="color:#6b7280">Click to inspect</span>` : null,
          ].filter(Boolean).join('<br/>');
          tooltip.html(lines);
        })
        .on('mousemove', (event) => {
          const rect = container.getBoundingClientRect();
          tooltip
            .style('left', `${event.clientX - rect.left + 12}px`)
            .style('top', `${event.clientY - rect.top - 10}px`);
        })
        .on('mouseleave', () => tooltip.style('opacity', '0'))
        .on('click', (event) => {
          if (!nodeClickRef.current) return;
          event.stopPropagation();
          nodeClickRef.current(node);
        });
    });

    drawRef.current = { g, x: xScale, y: yScale };

    return () => {
      tooltip.remove();
      drawRef.current = null;
    };
  }, [graph, activeNodeIds, activeEdgeIds, gradients, mode, minX, maxX, minY, maxY, layerLabels, interactive, layerInteractive]);

  // Selection overlay: redrawn alone so selecting never rebuilds thousands of edges.
  useEffect(() => {
    const draw = drawRef.current;
    if (!draw) return;
    draw.g.select('g.selection').remove();
    const sel = draw.g.append('g').attr('class', 'selection').attr('pointer-events', 'none');
    const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));

    if (selectedLayer !== null && selectedLayer !== undefined) {
      const ys = graph.nodes.filter((n) => n.layer === selectedLayer);
      if (ys.length) {
        const x = draw.x(ys[0].x);
        const top = Math.min(...ys.map((n) => draw.y(n.y))) - 20;
        const bottom = Math.max(...ys.map((n) => draw.y(n.y))) + 20;
        sel.append('rect')
          .attr('x', x - 24).attr('y', top).attr('width', 48).attr('height', bottom - top).attr('rx', 12)
          .attr('fill', 'rgba(253,224,71,0.06)').attr('stroke', SELECT_COLOR).attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '6,4');
      }
    }
    selectedEdgeIds?.forEach((i) => {
      const e = graph.edges[i];
      const s = e && nodeById.get(e.source);
      const t = e && nodeById.get(e.target);
      if (!s || !t) return;
      sel.append('line')
        .attr('x1', draw.x(s.x)).attr('y1', draw.y(s.y)).attr('x2', draw.x(t.x)).attr('y2', draw.y(t.y))
        .attr('stroke', SELECT_COLOR).attr('stroke-width', 3).attr('stroke-opacity', 0.95);
    });
    selectedNodeIds?.forEach((id) => {
      const n = nodeById.get(id);
      if (!n) return;
      const r = (n.layer_type === 'output' ? 14 : n.layer_type === 'input' ? 12 : 10) + 5;
      sel.append('circle')
        .attr('cx', draw.x(n.x)).attr('cy', draw.y(n.y)).attr('r', r)
        .attr('fill', 'none').attr('stroke', SELECT_COLOR).attr('stroke-width', 2.5);
    });
  }, [graph, selectedNodeIds, selectedEdgeIds, selectedLayer, mode, layerLabels, minX, maxX, minY, maxY, activeNodeIds, activeEdgeIds, gradients]);

  if (!graph.nodes.length) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-3">
        <div className="w-16 h-16 rounded-2xl bg-gray-800/50 flex items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="5" cy="12" r="2" /><circle cx="19" cy="5" r="2" /><circle cx="19" cy="19" r="2" />
            <line x1="7" y1="12" x2="17" y2="6" /><line x1="7" y1="12" x2="17" y2="18" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-gray-400">No network built yet</p>
          <p className="text-xs text-gray-600 mt-1">Configure and click "Build Network"</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <svg ref={svgRef} className="w-full h-full" />
      {mode !== 'signal' && (
        <div className="absolute bottom-3 left-3 flex flex-wrap gap-1.5">
          {Object.entries(LAYER_TYPE_COLORS)
            .filter(([type]) => graph.nodes.some((n) => n.layer_type === type))
            .map(([type, color]) => (
              <div key={type} className="flex items-center gap-1 text-xs text-gray-400 bg-gray-900/80 px-2 py-1 rounded-md border border-gray-800">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                <span className="capitalize">{type}</span>
              </div>
            ))}
        </div>
      )}
      {mode === 'backward' && (
        <div className="absolute top-3 right-3 flex items-center gap-2 text-xs bg-gray-900/90 px-2.5 py-1.5 rounded-lg border border-gray-800">
          {[['#3b82f6','vanishing'],['#10b981','healthy'],['#f59e0b','large'],['#ef4444','exploding']].map(([c,l]) => (
            <span key={l} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: c }} />
              <span style={{ color: 'var(--text-faint)' }}>{l}</span>
            </span>
          ))}
        </div>
      )}
      {mode !== 'backward' && showHint && (
        <div className="absolute top-3 right-3 text-xs bg-gray-900/80 px-2 py-1 rounded-md border border-gray-800" style={{ color: 'var(--text-faint)' }}>
          Scroll to zoom · Drag to pan
        </div>
      )}
    </div>
  );
}
