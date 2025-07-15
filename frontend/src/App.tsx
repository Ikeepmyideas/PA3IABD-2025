import React, { useState } from 'react';
import {
  Button,
  MenuItem,
  Select,
  TextField,
  Typography,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { LineChart } from '@mui/x-charts/LineChart';

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [model, setModel] = useState('linear');
  const [layers, setLayers] = useState([{ neurons: 8 }]);
  const [epochs, setEpochs] = useState(10);
  const [selectedGraph, setSelectedGraph] = useState<string[]>([]);
  const [resultsFile, setResultsFile] = useState<{
    results: { name: string; data: number[] }[];
  }>({ results: [] });

  const [resultsTraining, setResultsTraining] = useState<{ files: string[] }>({
    files: [],
  });

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImage(URL.createObjectURL(file));
  };

  const handleAddLayer = () => {
    setLayers([...layers, { neurons: 8 }]);
  };

  const handleChangeNeurons = (index: number, value: number) => {
    const updated = [...layers];
    updated[index].neurons = value;
    setLayers(updated);
  };

  const handleTrain = async () => {
    try {
      const nFeatures = 2; // à adapter dynamiquement plus tard
      const learningRate = 0.1;
      const activationType = 2;

      const xTrain = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
      ];
      const yTrain = [1.0, 0.0, 1.0, 0.0];

      // Crée le modèle linear
      const resCreate = await fetch(
        `http://localhost:8000/linear/create?n_features=${nFeatures}&learning_rate=${learningRate}&max_epochs=${epochs}&activation_type=${activationType}`,
        { method: 'POST' }
      );
      if (!resCreate.ok) throw new Error('Erreur création modèle');

      // Entraîne le modèle
      const resTrain = await fetch('http://localhost:8000/linear/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          x: xTrain,
          y: yTrain,
          show_logs: true,
        }),
      });
      if (!resTrain.ok) throw new Error('Erreur entraînement');

      // Pour la démo : on génère des pertes simulées
      const fakeLosses = Array.from({ length: epochs }, (_, i) =>
        parseFloat((1.0 / (i + 1)).toFixed(3))
      );
      const runName = `run_${Date.now()}`;

      setResultsTraining((prev) => ({
        files: [...prev.files, runName],
      }));

      setResultsFile((prev) => ({
        results: [...prev.results, { name: runName, data: fakeLosses }],
      }));

      setSelectedGraph((prev) => [...prev, runName]);
    } catch (err: any) {
      alert(`❌ ${err.message}`);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* Sidebar gauche */}
      <div style={{ width: '300px', background: '#f0f0f0', padding: '1rem' }}>
        <Typography variant="h6">📂 Importation</Typography>

        <Button
          component="label"
          variant="contained"
          startIcon={<CloudUploadIcon />}
          fullWidth
          sx={{ marginTop: '1rem' }}
        >
          Importer une image
          <input
            type="file"
            accept="image/*"
            hidden
            onChange={handleImageUpload}
          />
        </Button>

        <Typography variant="h6" sx={{ marginTop: '2rem' }}>
          🧠 Modèle
        </Typography>

        <Select
          fullWidth
          value={model}
          onChange={(e) => {
            setModel(e.target.value);
            setLayers([{ neurons: 8 }]);
          }}
        >
          <MenuItem value="linear">Linear</MenuItem>
          <MenuItem value="mlp">MLP</MenuItem>
          <MenuItem value="rbf">RBF</MenuItem>
          <MenuItem value="svm">SVM</MenuItem>
        </Select>

        {(model === 'mlp' || model === 'rbf') && (
          <div style={{ marginTop: '1rem' }}>
            <Typography variant="subtitle1">📐 Couches</Typography>
            {layers.map((layer, index) => (
              <TextField
                key={index}
                label={`Neurones couche ${index + 1}`}
                type="number"
                value={layer.neurons}
                fullWidth
                sx={{ marginBottom: '0.5rem' }}
                onChange={(e) =>
                  handleChangeNeurons(index, parseInt(e.target.value))
                }
              />
            ))}
            <Button
              variant="outlined"
              onClick={handleAddLayer}
              fullWidth
              sx={{ marginTop: '0.5rem' }}
            >
              ➕ Ajouter une couche
            </Button>
          </div>
        )}

        <Typography variant="h6" sx={{ marginTop: '2rem' }}>
          🔁 Époques
        </Typography>
        <TextField
          type="number"
          fullWidth
          value={epochs}
          onChange={(e) => setEpochs(parseInt(e.target.value))}
          inputProps={{ min: 1 }}
        />

        <Button
          variant="contained"
          color="primary"
          fullWidth
          sx={{ marginTop: '1.5rem' }}
          onClick={handleTrain}
        >
          🚀 Lancer l'entraînement
        </Button>
      </div>

      {/* Zone droite */}
      <div style={{ flex: 1, padding: '2rem' }}>
        <Typography variant="h5" gutterBottom>
          📊 Aperçu de l’image
        </Typography>
        {image ? (
          <img
            src={image}
            alt="importée"
            style={{ maxWidth: '100%', borderRadius: 8 }}
          />
        ) : (
          <Typography>Aucune image sélectionnée</Typography>
        )}

        <div className="view_training_results" style={{ marginTop: '3rem' }}>
          <Typography variant="h5">📈 Résultats d’entraînement</Typography>

          <div className="container-results" style={{ marginBottom: '1rem' }}>
            {resultsTraining?.files.map((file, index) => (
              <div key={index}>
                <Button
                  variant="contained"
                  sx={{
                    background: '#2874a6',
                    color: '#fff',
                    marginTop: '10px',
                    marginRight: '10px',
                  }}
                  onClick={() => {
                    setSelectedGraph((prev) =>
                      prev.includes(file)
                        ? prev.filter((f) => f !== file)
                        : [...prev, file]
                    );
                  }}
                >
                  {file}
                </Button>
              </div>
            ))}
          </div>

          <div className="container-charts">
            <LineChart
              xAxis={[
                {
                  data: Array.from(
                    {
                      length: Math.max(
                        ...(resultsFile?.results.map((r) => r.data.length) || [
                          1,
                        ])
                      ),
                    },
                    (_, i) => i + 1
                  ),
                  label: 'Epochs',
                },
              ]}
              series={
                resultsFile?.results
                  .filter((r) => selectedGraph.includes(r.name))
                  .map((r) => ({
                    data: r.data,
                    label: r.name,
                  })) || []
              }
              width={900}
              height={400}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
