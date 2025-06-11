// src/components/NeuralNetworkGraph.jsx
import React, { useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
} from "react-flow-renderer";

const layerSpacing = 200;
const nodeSize = 60;

const generateLayer = (count, layerIndex) =>
  Array.from({ length: count }).map((_, i) => ({
    id: `L${layerIndex}-N${i}`,
    data: { label: `N${i}` },
    position: {
      x: layerIndex * layerSpacing,
      y: i * (nodeSize + 40) + 50,
    },
    style: {
      background: "#ffffff",
      border: "2px solid #333",
      borderRadius: "50%",
      width: nodeSize,
      height: nodeSize,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
  }));

const connectLayers = (layer1, layer2) =>
  layer1.flatMap((from) =>
    layer2.map((to) => ({
      id: `e-${from.id}-${to.id}`,
      source: from.id,
      target: to.id,
      animated: true,
      style: { stroke: "#8884d8" },
    }))
  );

export default function NeuralNetworkGraph() {
  const [hiddenCount, setHiddenCount] = useState(4);

  const inputLayer = generateLayer(3, 0);
  const hiddenLayer = generateLayer(hiddenCount, 1);
  const outputLayer = generateLayer(2, 2);

  const nodes = [...inputLayer, ...hiddenLayer, ...outputLayer];
  const edges = [
    ...connectLayers(inputLayer, hiddenLayer),
    ...connectLayers(hiddenLayer, outputLayer),
  ];

  return (
    <div>
      <div style={{ marginBottom: 10, display: "flex", gap: 10 }}>
        <button onClick={() => setHiddenCount((n) => n + 1)}>
          ➕ Ajouter un neurone caché
        </button>
        <span>Neurones cachés : {hiddenCount}</span>
      </div>
      <div style={{ height: "500px", border: "1px solid #ccc", borderRadius: 10 }}>
        <ReactFlow nodes={nodes} edges={edges} fitView>
          <Background />
          <MiniMap />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}
