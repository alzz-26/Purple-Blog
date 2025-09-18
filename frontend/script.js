document.getElementById('predictBtn').onclick = async () => {
  // Get values from all input fields
  const description = document.getElementById('description').value;
  const followers = document.getElementById('followers').value;
  const views = document.getElementById('views').value; // Get views

  if (!description || !followers || !views) {
    alert('Please fill out all fields.');
    return;
  }

  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = 'Analyzing...';

  // Send all three inputs in the JSON body 
  const res = await fetch('http://127.0.0.1:5000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      description: description, 
      followers: parseInt(followers),
      views: parseInt(views) // Add views
    })
  });

  const data = await res.json();

  if (data.error) {
    resultDiv.innerHTML = `<p style="color: #ffcccc;">Error: ${data.error}</p>`;
  } else {
    resultDiv.innerHTML = `
      <p><strong>Predicted Likes:</strong> ${data.predicted_likes}</p>
      <p><strong>Predicted Comments:</strong> ${data.predicted_replies}</p>
    `;
  }
};