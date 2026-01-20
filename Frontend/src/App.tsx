import { BrowserRouter, Routes, Route } from "react-router-dom"
import Nav from "./components/Nav"
import Find from "./pages/Find"
import Profile from "./pages/Profile"
import Login from "./pages/Login"

export default function App() {
  return (
    <BrowserRouter>
      <Nav />

      <Routes>
        <Route path="/" element={<Find />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/login" element={<Login />} />
      </Routes>
    </BrowserRouter>
  )
}
