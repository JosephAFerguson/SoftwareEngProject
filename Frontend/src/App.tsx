import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom"
import { GoogleMapsProvider } from "./components/GoogleMapsProvider"
import Nav from "./components/Nav"
import Find from "./pages/Find"
import Host from "./pages/Host"
import Profile from "./pages/Profile"
import Login from "./pages/Login"
import Signup from "./pages/SignUp"

function Layout() {
  const location = useLocation()
  const hideNav = location.pathname === "/login" || location.pathname === "/signup"

  return (
    <>
      {!hideNav && <Nav />}
      <GoogleMapsProvider>
        <Routes>
          <Route path="/" element={<Find />} />
          <Route path="/host" element={<Host />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
        </Routes>
      </GoogleMapsProvider>
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
