import { Link } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./Nav.module.css"

export default function Nav() {
  const {signedIn} = useAppStore()

  return (
    <nav className={styles.nav}>
      <div className={styles.centerLinks}>
        <Link className={styles.link} to="/">
          Find
        </Link>

        <Link className={`${styles.link} ${styles.logoLink}`} to="/">
          SubleaseInc.
        </Link>


        <Link className={styles.link} to="/host">
          Host
        </Link>
      </div>

        {signedIn ? (
          <Link className={styles.profileLink} to="/profile">Profile</Link>
        ) : (
          <Link className={styles.profileLink} to="/login">Login</Link>
        )}
    </nav>
  )
}
