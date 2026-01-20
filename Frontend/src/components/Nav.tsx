import { Link } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./Nav.module.css"
import logo from  "../../public/logo.svg"

export default function Nav() {
  const {signedIn} = useAppStore()

  return (
    <nav className={styles.nav}>
      <div className={styles.centerLinks}>
        <Link className={styles.link} to="/">
          Find
        </Link>

        <Link className={styles.link} to="/host">
          Host
        </Link>

        <img className={styles.logo}
          src={logo}
          alt="App Logo"
        />
      </div>

        {signedIn ? (
          <Link className={styles.profileLink} to="/profile">Profile</Link>
        ) : (
          <Link className={styles.profileLink} to="/login">Login</Link>
        )}
    </nav>
  )
}
