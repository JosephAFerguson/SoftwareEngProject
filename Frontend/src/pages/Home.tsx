export default function Home() {
  return (
    <main>
      <h1>Welcome to Subleasers.inc</h1>
      <p>Find or host subleases with ease.</p>
    </main>
  )
}

/* Example function to fetch recommendations from the backend API 
async function getRecommendations() {
  const res = await fetch("http://localhost:5000/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      location: "Boston",
      price_max: 1200,
      bedrooms: 2
    })
  })

  const data = await res.json()
  console.log(data)
}
*/
