from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class BetInfoStruct:
    position: int
    amount: Decimal
    claimed: bool

@dataclass
class BetInfoStructOutput:
    position: int
    amount: Decimal
    claimed: bool

@dataclass
class DogeBetsPrediction:
    def Bets(self, epoch: int, user: str) -> BetInfoStructOutput:
        pass

    def BlackListInsert(self, user_address: str) -> None:
        pass

    def BlackListRemove(self, user_address: str) -> None:
        pass

    def ChangePriceSource(self, new_price_source: str) -> None:
        pass

    def Claimable(self, epoch: int, user: str) -> bool:
        pass

    def Execute(self, price: int, timestamp: int, bet_on_bull: int, bet_on_bear: int) -> None:
        pass

    def FundsExtract(self, value: int) -> None:
        pass

    def FundsInject(self) -> None:
        pass

    def GetTotalReservedReferralFunds(self) -> Decimal:
        pass

    def GetUserRounds(self, user: str, cursor: int, size: int) -> Tuple[List[int], List[BetInfoStructOutput], int]:
        pass

    def GetUserRoundsLength(self, user: str) -> int:
        pass

    def HouseBet(self, bull_amount: int, bear_amount: int) -> None:
        pass

    def HouseBetsWithinLimits(self, bet_bull: int, bet_bear: int) -> bool:
        pass

    def IsPaused(self) -> bool:
        pass

    def OwnershipRenounce(self) -> None:
        pass

    def OwnershipTransfer(self, new_owner: str) -> None:
        pass

    def Pause(self) -> None:
        pass

    def ReferralRewardsAvailable(self, user: str) -> Decimal:
        pass

    def Refundable(self, epoch: int, user: str) -> bool:
        pass

    def RewardUser(self, user: str, value: int) -> None:
        pass

    def RoundCancel(self, epoch: int, canceled: bool, closed: bool) -> None:
        pass

    def RoundLock(self, price: int, timestamp: int) -> None:
        pass

    def RoundStart(self) -> None:
        pass

    def Rounds(self, epoch: int) -> Tuple[int, int, int, int, int, int, int, int, int, int, int, int, bool, bool]:
        pass

    def SetHouseBetMinRatio(self, min_bear_to_bull_ratio_percents: int) -> None:
        pass

    def SetMinBetAmount(self, new_min_bet_amount: int) -> None:
        pass

    def SetOperator(self, operator_address: str) -> None:
        pass

    def SetReferralsContract(self, new_contract_address: str) -> None:
        pass

    def SetRewardRate(self, new_reward_rate: int) -> None:
        pass

    def SetRoundBufferAndInterval(self, round_buffer_seconds: int, round_interval_seconds: int) -> None:
        pass

    def Unpause(self) -> None:
        pass

    def UserBets(self, user: str, epoch: int) -> int:
        pass

    def currentBlockNumber(self) -> int:
        pass

    def currentBlockTimestamp(self) -> int:
        pass

    def currentEpoch(self) -> int:
        pass

    def currentSettings(self) -> Tuple[bool, bool, bool, int, int, str, int]:
        pass

    def lockedOnce(self) -> bool:
        pass

    def minBetAmount(self) -> int:
        pass

    def minimumRewardRate(self) -> int:
        pass

    def operatorAddress(self) -> str:
        pass

    def owner(self) -> str:
        pass

    def priceSource(self) -> str:
        pass

    def referralsContract(self) -> str:
        pass

    def rewardRate(self) -> int:
        pass

    def roundBuffer(self) -> int:
        pass

    def roundInterval(self) -> int:
        pass

    def startedOnce(self) -> bool:
        pass

    def user_BetBear(self, epoch: int) -> None:
        pass

    def user_BetBearSpecial(self, epoch: int, new_referrer: str) -> None:
        pass

    def user_BetBull(self, epoch: int) -> None:
        pass

    def user_BetBullSpecial(self, epoch: int, new_referrer: str) -> None:
        pass

    def user_Claim(self, epochs: List[int]) -> None:
        pass

    def user_ReferralFundsClaim(self) -> None:
        pass
