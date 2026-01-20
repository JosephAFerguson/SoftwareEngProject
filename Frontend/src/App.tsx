import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom"
import Nav from "./components/Nav"
import Find from "./pages/Find"
import Profile from "./pages/Profile"
import Login from "./pages/Login"
import Signup from "./pages/SignUp"

function Layout() {
  const location = useLocation()
  const hideNav = location.pathname === "/login"

  return (
    <>
      {!hideNav && <Nav />}

      <Routes>
        <Route path="/" element={<Find />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
      </Routes>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  )
}
