import React, { useState, useEffect } from "react";
import ReactFlow, { addEdge, Background, Controls } from "react-flow-renderer";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import NeuralNetworkGraph from "./NeuralNetworkGraph";

const initialNodes = [
  { id: "1", data: { label: "Entrée" }, position: { x: 250, y: 5 } },
];
const initialEdges = [];

export default function App() {
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  const [lossHistory, setLossHistory] = useState([]);
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [rawOutput, setRawOutput] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [modelType, setModelType] = useState("linear");
  const [variant, setVariant] = useState("tanh");
  const [testResults, setTestResults] = useState(null);

  const onConnect = (params) => setEdges((eds) => addEdge(params, eds));

  useEffect(() => {
    fetch("http://localhost:8000/load-model", { method: "POST" })
      .then(() => console.log("Modèle chargé"))
      .catch(() => alert("Erreur lors du chargement du modèle"));
  }, []);

  const addNode = () => {
    const id = (nodes.length + 1).toString();
    const newNode = {
      id,
      data: { label: `Nœud ${id}` },
      position: {
        x: 100 + Math.random() * 400,
        y: 100 + Math.random() * 200,
      },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(URL.createObjectURL(file));
    const formData = new FormData();
    formData.append("file", file);

    let url = `http://localhost:8000/predict?model=${modelType}`;
    if (modelType === "linear") {
      url += `&variant=${variant}`;
    }

    try {
      const res = await fetch(url, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.error) {
        alert(`❌ Erreur : ${data.error}`);
        setPrediction(null);
        setRawOutput(null);
        setAccuracy(null);
        return;
      }

      setPrediction(typeof data.prediction === "number" ? data.prediction : null);
      setRawOutput(typeof data.raw_output === "number" ? data.raw_output : null);
      setAccuracy(typeof data.accuracy === "number" ? data.accuracy : null);
    } catch (error) {
      console.error("Erreur lors de la prédiction :", error);
      alert("❌ Erreur de communication avec le serveur.");
    }
  };

  const fetchLoss = async () => {
    try {
      const res = await fetch("http://localhost:8000/loss-history");
      const data = await res.json();
      setLossHistory(data.history);
    } catch (error) {
      console.error("Erreur lors du chargement de la courbe :", error);
    }
  };

  const retrainModel = async () => {
    try {
      await fetch(`http://localhost:8000/train?model=${modelType}`, { method: "POST" });
      alert("✅ Modèle réentraîné avec succès.");
    } catch {
      alert("❌ Erreur lors du réentraînement.");
    }
  };

  const saveModel = async () => {
    try {
      await fetch("http://localhost:8000/save-model", { method: "POST" });
      alert("💾 Modèle sauvegardé.");
    } catch {
      alert("❌ Erreur lors de la sauvegarde.");
    }
  };

  const loadModel = async () => {
    try {
      await fetch("http://localhost:8000/load-model", { method: "POST" });
      alert("📂 Modèle chargé.");
    } catch {
      alert("❌ Erreur lors du chargement.");
    }
  };

  const testOnDataset = async () => {
    try {
      const res = await fetch("http://localhost:8000/test-dataset");
      const data = await res.json();
      if (data.error) {
        alert(`❌ Erreur : ${data.error}`);
        return;
      }
      setTestResults(data);
    } catch (error) {
      alert("❌ Erreur lors du test du dataset");
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      <div style={{ width: "70%", padding: "1rem" }}>
        <h3>🧠 Réseau de Neurones (visuel)</h3>
        <NeuralNetworkGraph />

        <h3 style={{ marginTop: "2rem" }}>📊 Graphe interactif</h3>
        <button onClick={addNode} style={{ marginBottom: "1rem" }}>
          ➕ Ajouter un nœud
        </button>
        <ReactFlow nodes={nodes} edges={edges} onConnect={onConnect} fitView>
          <Background />
          <Controls />
        </ReactFlow>
      </div>

      <div style={{ width: "30%", padding: "1rem", overflowY: "auto", background: "#f8f8f8" }}>
        <section>
          <h3>🖼️ Prédiction par image</h3>

          <div style={{ marginBottom: 10 }}>
            <strong>Modèle choisi :</strong>
            <div style={{ display: "flex", gap: "0.5rem", marginTop: 5, flexWrap: "wrap" }}>
              {["linear", "mlp", "rbfn", "svm"].map((model) => (
                <button
                  key={model}
                  onClick={() => setModelType(model)}
                  style={{
                    backgroundColor: modelType === model ? "#d0eaff" : "#fff",
                    border: "1px solid #ccc",
                    padding: "0.4rem",
                    borderRadius: "4px",
                  }}
                >
                  {model === "linear"
                    ? "🔷 Linear"
                    : model === "mlp"
                    ? "🔶 MLP"
                    : model === "rbfn"
                    ? "🔺 RBFN"
                    : "📐 SVM"}
                </button>
              ))}
            </div>
            <p style={{ marginTop: 5 }}>
              <em>Modèle actuel : {modelType.toUpperCase()}</em>
            </p>
          </div>

          {modelType === "linear" && (
            <div style={{ marginTop: 10 }}>
              <strong>⚙️ Variante :</strong>
              <div style={{ display: "flex", gap: "0.5rem", marginTop: 5 }}>
                {["none", "tanh", "sigmoid"].map((v) => (
                  <button
                    key={v}
                    onClick={() => setVariant(v)}
                    style={{
                      backgroundColor: variant === v ? "#ffdcdc" : "#fff",
                      border: "1px solid #ccc",
                      padding: "0.3rem",
                      borderRadius: "4px",
                    }}
                  >
                    {v === "none" ? "Régression" : v === "tanh" ? "Tanh (binaire)" : "Sigmoid (multi)"}
                  </button>
                ))}
              </div>
              <p style={{ marginTop: 5 }}>
                <em>Variante : {variant}</em>
              </p>
            </div>
          )}

          <input type="file" accept="image/*" onChange={handleImageUpload} />
          {image && (
            <img
              src={image}
              alt="Aperçu"
              style={{ width: "100%", marginTop: 10, borderRadius: 4 }}
            />
          )}

          {prediction !== null && !isNaN(prediction) && prediction >= 0 && prediction <= 25 ? (
            <p>
              🔤 Prédiction : <strong>{String.fromCharCode(65 + prediction)}</strong>
            </p>
          ) : prediction !== null ? (
            <p>🔤 Prédiction : <strong>{prediction}</strong></p>
          ) : null}

          {rawOutput !== null && (
            <p>
              🔢 Score brut : <strong>{rawOutput.toFixed(4)}</strong>
            </p>
          )}

          {accuracy !== null && (
            <p>
              📊 Précision : <strong>{accuracy.toFixed(2)}%</strong>
            </p>
          )}
        </section>

        <section style={{ marginTop: 30 }}>
          <h3>📉 Historique de la loss</h3>
          <button onClick={fetchLoss}>📈 Afficher la courbe</button>
          {lossHistory.length > 0 && (
            <LineChart width={300} height={200} data={lossHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="loss" stroke="#8884d8" />
            </LineChart>
          )}
        </section>

        <section style={{ marginTop: 30 }}>
          <h3>🧠 Gestion du modèle</h3>
          <button onClick={retrainModel}>🔁 Réentraîner</button>
          <button onClick={saveModel}>💾 Sauvegarder</button>
          <button onClick={loadModel}>📂 Charger</button>
          <button onClick={testOnDataset}>📊 Tester sur le dataset</button>

          {testResults && (
            <div style={{ marginTop: 10 }}>
              <p>✅ Résultats :</p>
              <ul>
                {Object.entries(testResults).map(([model, res]) => (
                  <li key={model}>
                    {model.toUpperCase()} : {res?.accuracy ? `${res.accuracy.toFixed(2)}%` : "❌ Échec"}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
