import dotenv from 'dotenv'

import { PLATFORMS, startPolling } from './lib'

dotenv.config()

startPolling(
  process.env.PRIVATE_KEY,
  process.env.BET_AMOUNT,
  PLATFORMS.PancakeSwap
).catch((error) => {
  console.error(error)
})
