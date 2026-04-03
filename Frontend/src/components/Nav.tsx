import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./Nav.module.css"

const RENTAL_API_URL = "http://localhost:3000/api/v1/rental"

type RentalSummary = {
  user_id: number
}

export default function Nav() {
  const { signedIn, userId } = useAppStore()
  const [hasListing, setHasListing] = useState(false)

  useEffect(() => {
    const syncListingStatus = async () => {
      if (!signedIn || !userId) {
        setHasListing(false)
        return
      }

      try {
        const response = await fetch(`${RENTAL_API_URL}/all`)
        if (!response.ok) {
          setHasListing(false)
          return
        }

        const data: { rentals?: RentalSummary[] } = await response.json()
        setHasListing((data.rentals ?? []).some((r) => r.user_id === userId))
      } catch {
        setHasListing(false)
      }
    }

    syncListingStatus()
    window.addEventListener("listing-status-changed", syncListingStatus)

    return () => {
      window.removeEventListener("listing-status-changed", syncListingStatus)
    }
  }, [signedIn, userId])

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
          {hasListing ? "Your Listing" : "Host"}
        </Link>
      </div>

      {signedIn ? (
        <div className={styles.rightLinks}>
          <Link className={styles.inboxLink} to="/inbox" aria-label="Inbox" title="Inbox">
            {"\u{1F4E5}"}
          </Link>
          <Link className={styles.profileLink} to="/profile">Profile</Link>
        </div>
      ) : (
        <Link className={styles.profileLink} to="/login">Login</Link>
      )}
    </nav>
  )
}
