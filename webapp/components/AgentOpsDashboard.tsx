"use client";

import React, { useEffect, useMemo, useState } from "react";
import ReactFlow, { Background, Controls, Edge, Node, Position } from "reactflow";
import "reactflow/dist/style.css";

// -----------------------------
// Types
// -----------------------------

type StepType = "prompt" | "thinking" | "tool" | "output";

type Step = {
  id: string;
  type: StepType;
  name: string;
  created_at?: string; // ISO
  duration?: number; // seconds
  [key: string]: any;
};

type Task = {
  id: string;
  name: string;
  steps: Step[];
};

type Run = {
  id: string;
  tasks: Task[];
};

// -----------------------------
// Utility helpers
// -----------------------------

const stepColor: Record<StepType, string> = {
  prompt: "bg-green-200 border-green-400 text-green-900",
  thinking: "bg-green-100 border-green-300 text-green-900",
  tool: "bg-blue-200 border-blue-400 text-blue-900",
  output: "bg-green-200 border-green-400 text-green-900",
};

function fmtTime(s?: string) {
  if (!s) return "";
  const d = new Date(s);
  return d.toLocaleTimeString();
}

function fmtDuration(n?: number) {
  if (n == null) return "";
  return `${n.toFixed(0)}s`;
}

// Flatten steps with task context
function flatten(run?: Run) {
  if (!run) return [] as { step: Step; task: Task; taskIndex: number; stepIndex: number }[];
  const out: { step: Step; task: Task; taskIndex: number; stepIndex: number }[] = [];
  run.tasks.forEach((t, ti) => t.steps.forEach((s, si) => out.push({ step: s, task: t, taskIndex: ti, stepIndex: si })));
  return out;
}

// -----------------------------
// Main component
// -----------------------------

export default function AgentOpsDashboard() {
  const [runs, setRuns] = useState<string[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [selected, setSelected] = useState<{ step: Step; task: Task; taskIndex: number; stepIndex: number } | null>(null);

  // Load list of runs once
  useEffect(() => {
    fetch('http://localhost:8000/runs', {method: 'GET'})
    .then((r) => r.json())
    .then((data) => {
      const run_list = (data as { runs: string[] }).runs;
      setRuns(run_list);
      if (run_list.length > 0) {
        setSelectedRunId(run_list[0]);
      }
    });
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    setRun(null);
    setSelected(null);

    fetch('http://localhost:8000/run/' + selectedRunId, {method: 'GET'})
    .then((r) => r.json())
    .then((data) => {
      const runDetails = data as Run;
      setRun(runDetails);
      setSelectedRunId(runDetails.id);
    });
  }, [selectedRunId]);

  const flat = useMemo(() => flatten(run || undefined), [run]);

  // Build React Flow graph
  const graph = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];

    if (!run) return { nodes, edges };

    const x = 0; // single vertical lane like the sketch
    let y = 0;
    const PAD_Y = 24;
    const STEP_H = 60;

    run.tasks.forEach((task, ti) => {
      // Task label (simple text node)
      nodes.push({
        id: `task-${task.id}`,
        position: { x, y },
        data: { label: task.name },
        type: "input",
        style: { fontWeight: 600 },
      });
      y += STEP_H;

      task.steps.forEach((s, si) => {
        const id = s.id;
        nodes.push({
          id,
          position: { x, y },
          data: {
            label: (
              <div className={`flex items-center justify-between gap-3 px-3 py-2 border rounded-xl ${stepColor[s.type]}`}>
                <span className="font-medium">{`[${s.type}] ${s.name}`}</span>
                <div className="text-xs opacity-80 flex gap-2">
                  <span>{fmtTime(s.created_at)}</span>
                  {s.hasOwnProperty("duration") && (
                    <span>· {fmtDuration(s.duration)}</span>
                  )}
                  <span className="text-xs text-gray-500"></span>
                </div>
              </div>
            ),
          },
          sourcePosition: Position.Bottom,
          targetPosition: Position.Top,
          draggable: false,
          selectable: true,
          style: { width: 360, padding: 0, border: 'none' },
        });

        // Edge from previous
        const prevId = si === 0 ? `task-${task.id}` : task.steps[si - 1].id;
        edges.push({ id: `${prevId}->${id}`, source: prevId, target: id, animated: true });

        y += STEP_H;
      });

      // Spacer between tasks
      y += PAD_Y;
    });

    return { nodes, edges };
  }, [run]);

  const getHistogramColor = (stepType: StepType, isSelected: boolean) => {
    const colors = {
      thinking: isSelected ? "bg-green-300 border-green-500" : "bg-green-100 border-green-300",
      tool: isSelected ? "bg-blue-500 border-blue-700" : "bg-blue-200 border-blue-400",
      output: isSelected ? "bg-green-500 border-green-700" : "bg-green-200 border-green-400",
      prompt: isSelected ? "bg-green-500 border-green-700" : "bg-green-200 border-green-400", // fallback
    };
    return colors[stepType];
  };

  // Histogram data (all non-prompt steps in the run)
  const histogram = useMemo(() => {
    const items = flat.filter((f) => f.step.type !== "prompt");
    const max = Math.max(1, ...items.map((i) => i.step.duration || 0));
    return items.map((i) => ({
      key: i.step.id,
      h: ((i.step.duration || 0) / max) * 100,
      duration: i.step.duration || 0,
      label: i.step.title || i.step.type,
      type: i.step.type, // Add step type
      ti: i.taskIndex,
    }));
  }, [flat]);

  // Progress lines for the selected step
  const progress = useMemo(() => {
    if (!selected || !run) return { inTask: 0, inRun: 0 };
    const { taskIndex, stepIndex } = selected;
    const task = run.tasks[taskIndex];
    const steps = task.steps.filter((s) => s.type !== "prompt");
    const selIdx = Math.max(0, selected.step.type === "prompt" ? 0 : steps.findIndex((s) => s.id === selected.step.id) + 1);

    const inRun = (taskIndex + 1) / run.tasks.length; // advancement in # of task

    // time in task (seconds)
    const total = steps.reduce((a, s) => a + (s.duration || 0), 0);
    const before = steps.slice(0, Math.max(0, selIdx)).reduce((a, s) => a + (s.duration || 0), 0);
    const inTask = total ? before / total : 0;

    return { inTask, inRun };
  }, [selected, run]);

  return (
    <div className="h-screen w-screen grid grid-cols-[240px_1fr_360px] grid-rows-[1fr_230px] gap-0 text-sm text-black">
      {/* left panel : list of runs */}
      <aside className="col-start-1 row-span-2 border-r bg-white">
        <div className="px-4 py-3 font-semibold">Runs</div>
        <div className="space-y-2 px-3 pb-4 overflow-auto h-[calc(100%-44px)]">
          {runs.map((run_id) => (
            <button
              key={run_id}
              onClick={() => setSelectedRunId(run_id)}
              className={`w-full text-left px-3 py-2 rounded-lg border ${
                selectedRunId === run_id ? "bg-green-100 border-green-400" : "bg-white hover:bg-gray-50"
              }`}
            >
              {run_id}
            </button>
          ))}
        </div>
      </aside>

      {/* center: execution flow */}
      <main className="col-start-2 row-start-1 row-end-2 border-r">
        {!run ? (
          <div className="h-full flex items-center justify-center text-gray-500">Loading run…</div>
        ) : (
          <div className="h-full bg-white">
            <ReactFlow
              nodes={graph.nodes}
              edges={graph.edges}
              fitView
              proOptions={{ hideAttribution: true }}
              onNodeClick={(_, node) => {
                const found = flat.find((f) => f.step.id === node.id);
                if (found) setSelected(found);
              }}
            >
              <Background />
              <Controls />
            </ReactFlow>
          </div>
        )}
      </main>

      {/* right panel: step details */}
      <aside className="col-start-3 row-start-1 row-end-2 bg-white">
        <div className="px-4 py-3 font-semibold border-b">
          {selected?.step.type ? (
            <div className="flex items-center justify-between">
              <span>
                {selected.step.name}
              </span>
              <span className="text-xs text-gray-500">
                {selected.step.hasOwnProperty("created_at") && (
                  <>
                    {fmtTime(selected.step.created_at)}
                  </>
                )}
                {selected.step.hasOwnProperty("duration") && (
                  <>
                    · {fmtDuration(selected.step.duration)}
                  </>
                )}
              </span>
            </div>
          ) : (
            "Step details"
          )}
        </div>
        <div className="p-4 space-y-3 overflow-auto h-[calc(100%-48px)] max-h-[calc(100vh-278px)]">
          {!selected ? (
            <div className="text-gray-500">Click any step in the graph to see details.</div>
          ) : (
            <>
              {Object.entries(selected.step)
                .filter(([key]) => !['id', 'type', 'name', 'created_at', 'duration'].includes(key))
                .map(([key, value]) => (
                  <div key={key}>
                    <div className="text-xs uppercase text-gray-500">{key}</div>
                    <pre className="bg-gray-50 p-3 rounded border text-[12px] overflow-auto">
                      {JSON.stringify(value, null, 2)}
                    </pre>
                  </div>
                ))}
            </>
          )}
        </div>
      </aside>

      {/* footer */}
      <section className="col-start-2 col-end-4 row-start-2 border-t bg-white">
        {/* histogram */}
        <div className="px-4 py-2 text-xs text-gray-600">Steps duration</div>
        <div className="px-4 pb-3">
          <div className="h-24 flex items-end gap-2">
            {histogram.map((b) => (
              <div key={b.key} className="flex flex-col items-center" title={`${b.label} • ${b.duration}s`}>
                <div
                  className={`w-5 rounded-t border ${getHistogramColor(b.type, selected?.step.id === b.key)}`}
                  style={{ height: `${Math.max(10, (b.h / 100) * 96)}px` }}
                />
                <div className="text-[10px] mt-1 text-gray-500">{b.duration.toFixed(0)}s</div>
              </div>
            ))}
          </div>

          {/* progress bars */}
          <div className="mt-3 space-y-2">
            <div className="text-xs text-gray-600">Advancement in task (time)</div>
            <div className="h-2 rounded bg-gray-100 overflow-hidden">
              <div 
                className="h-full bg-green-400 transition-all duration-300 ease-out" 
                style={{ width: `${Math.round(progress.inTask * 100)}%` }} 
              />
            </div>
            <div className="text-xs text-gray-600">Advancement in run (# of tasks)</div>
            <div className="h-2 rounded bg-gray-100 overflow-hidden">
              <div 
                className="h-full bg-green-600 transition-all duration-300 ease-out" 
                style={{ width: `${Math.round(progress.inRun * 100)}%` }} 
              />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
