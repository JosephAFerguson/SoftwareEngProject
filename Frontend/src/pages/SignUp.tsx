import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./Signup.module.css"

const SIGNUP_API_URL = "http://localhost:3000/api/v1/auth/signup"

export default function Signup() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMessage, setErrorMessage] = useState("")

  const setSignedIn = useAppStore(state => state.setSignedIn)
  const setUserId = useAppStore(state => state.setUserId)
  const navigate = useNavigate()

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setErrorMessage("")

    if (password !== confirmPassword) {
      setErrorMessage("Passwords do not match.")
      return
    }

    setIsSubmitting(true)

    try {
      const response = await fetch(SIGNUP_API_URL, {
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
        throw new Error(body?.error || "Failed to sign up.")
      }

      const body = await response.json().catch(() => ({}))
      const userId = typeof body?.user_id === "number" ? body.user_id : null
      if (!userId) {
        throw new Error("Signup succeeded but no user_id was returned.")
      }

      setSignedIn(true)
      setUserId(userId)
      navigate("/profile")
    } catch (error) {
      setErrorMessage((error as Error).message || "Failed to sign up.")
    } finally {
      setIsSubmitting(false)
    }

  }

  return (
    <div className={styles.signupContainer}>
      <h1 className={styles.header}>Sign Up</h1>
      {errorMessage && <p>{errorMessage}</p>}

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

        <input
          type="password"
          placeholder="Confirm Password"
          value={confirmPassword}
          onChange={e => setConfirmPassword(e.target.value)}
          required
        />

        <button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Creating Account..." : "Sign Up"}
        </button>
      </form>
    </div>
  )
}
