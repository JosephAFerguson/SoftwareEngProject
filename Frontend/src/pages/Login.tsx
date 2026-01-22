import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./Login.module.css"

export default function Login() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  const setSignedIn = useAppStore(state => state.setSignedIn)
  const navigate = useNavigate()

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()

    //TODO send to backend & validate


    setSignedIn(true)
    navigate("/find")
  }

  return (
    <div className={styles.loginContainer}>
      <h1 className={styles.header}>Login</h1>

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

        <button type="submit">Login</button>
      </form>

      <p>New to SubleasesInc? <a href="/signup">Sign up</a></p>
    </div>
  )
}
