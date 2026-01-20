import { Link } from "react-router-dom"
import styles from "./Nav.module.css"

export default function Nav() {
  return (
    <nav className={styles.nav}>
      <Link to="/">
        Home
      </Link>

      <Link to="/profile">
        Profile
      </Link>

      <Link to="/login">
        Login
      </Link>
    </nav>
  )
}
