const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

app.post('/api/predict', (req, res) => {
  // The request body should be an array of 280 numeric features
  const features = req.body;

  const modelDir = path.join(__dirname, 'model');
  const scriptPath = path.join(modelDir, 'predict.py');

  const py = spawn('python', [scriptPath], {cwd: modelDir});

  py.stdin.write(JSON.stringify(features));
  py.stdin.end();

  let stdout = '';
  let stderr = '';

  py.stdout.on('data', chunk => stdout += chunk);
  py.stderr.on('data', chunk => stderr += chunk);

  py.on('close', code => {
    if (code !== 0) {
      console.error(`Python error (code ${code}): ${stderr}`);
      res.status(500).json({ error: stderr || 'Python script error' });
      return;
    }
    try {
      res.json(JSON.parse(stdout));
    } catch (e) {
      res.status(500).json({ error: 'Invalid JSON from Python' });
    }
  });
});

app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
