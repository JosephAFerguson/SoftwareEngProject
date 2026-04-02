import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./Login.module.css"

const LOGIN_API_URL = "http://localhost:3000/api/v1/auth/login"
const UC_EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@(mail\.uc\.edu|uc\.edu)$/

export default function Login() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMessage, setErrorMessage] = useState("")

  const setSignedIn = useAppStore(state => state.setSignedIn)
  const setUserId = useAppStore(state => state.setUserId)
  const navigate = useNavigate()

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setErrorMessage("")

    if (!UC_EMAIL_REGEX.test(email.trim())) {
      setErrorMessage("Please use your UC email address (@mail.uc.edu or @ucmail.uc.edu).")
      return
    }

    setIsSubmitting(true)
    try {
      const response = await fetch(LOGIN_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: email.trim().toLowerCase(),
          password,
        }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body?.error || "Failed to log in.")
      }
      const body = await response.json().catch(() => ({}))
      const userId = typeof body?.user_id === "number" ? body.user_id : null
      if (!userId) {
        throw new Error("Login succeeded but no user_id was returned.")
      }
      setSignedIn(true)
      setUserId(userId)
      navigate("/")
    } catch (error) {
      setErrorMessage((error as Error).message || "Failed to log in.")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className={styles.loginContainer}>
      <div className={styles.card}>
        <div className={styles.logoRow}>
          <span className={styles.logoIcon}>🏠</span>
          <span className={styles.logoText}>SubleasesInc</span>
        </div>
        <h1 className={styles.header}>Welcome back</h1>
        <p className={styles.subheader}>Sign in with your UC email</p>

        {errorMessage && <p className={styles.errorText}>{errorMessage}</p>}

        <form className={styles.form} onSubmit={handleSubmit}>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>UC Email</label>
            <input
              type="email"
              placeholder="m12345@mail.uc.edu"
              value={email}
              onChange={e => setEmail(e.target.value)}
              className={styles.input}
              required
            />
          </div>

          <div className={styles.fieldGroup}>
            <label className={styles.label}>Password</label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className={styles.input}
              required
            />
          </div>

          <button type="submit" disabled={isSubmitting} className={styles.submitBtn}>
            {isSubmitting ? "Logging In..." : "Log In"}
          </button>
        </form>

        <p className={styles.switchText}>
          New to SubleasesInc? <a href="/signup" className={styles.link}>Sign up</a>
        </p>
      </div>
    </div>
  )
}
