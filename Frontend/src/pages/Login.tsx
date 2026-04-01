import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./Login.module.css"

const LOGIN_API_URL = "http://localhost:3000/api/v1/auth/login"

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
      <h1 className={styles.header}>Login</h1>
      {errorMessage && <p className={styles.errorText}>{errorMessage}</p>}

      <form className={styles.form} onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />

        <button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Logging In..." : "Login"}
        </button>
      </form>

      <p>New to SubleasesInc? <a href="/signup">Sign up</a></p>
    </div>
  )
}
