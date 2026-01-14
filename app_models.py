"""
Pydantic models for app state and API schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ==================== Amazon ====================
class AmazonProduct(BaseModel):
    product_id: str
    name: str
    price: float
    category: str
    rating: float
    
class AmazonOrder(BaseModel):
    order_id: str
    product_id: str
    product_name: str
    quantity: int
    total_price: float
    order_date: datetime
    
class AmazonState(BaseModel):
    user_id: str
    prime_member: bool = False
    order_history: List[AmazonOrder] = Field(default_factory=list)
    search_history: List[str] = Field(default_factory=list)  # search queries
    viewed_products: List[str] = Field(default_factory=list)  # product_ids
    cart: List[Dict[str, Any]] = Field(default_factory=list)  # {product_id, quantity}
    wishlist: List[str] = Field(default_factory=list)  # product_ids

# ==================== Spotify ====================
class SpotifyPlaylist(BaseModel):
    playlist_id: str
    name: str
    song_ids: List[str] = Field(default_factory=list)
    
class SpotifySong(BaseModel):
    song_id: str
    title: str
    artist: str
    genre: str
    duration_seconds: int
    
class SpotifyPlayHistory(BaseModel):
    song_id: str
    played_at: datetime
    duration_played: int  # seconds actually played
    
class SpotifyState(BaseModel):
    user_id: str
    premium: bool = False
    playlists: List[SpotifyPlaylist] = Field(default_factory=list)
    followed_artists: List[str] = Field(default_factory=list)  # artist_ids
    play_history: List[SpotifyPlayHistory] = Field(default_factory=list)
    favorite_genres: List[str] = Field(default_factory=list)
    
# ==================== Fitbit ====================
class FitbitGoal(BaseModel):
    goal_type: str  # "steps", "active_minutes", "weight", etc.
    target_value: float
    
class FitbitWorkout(BaseModel):
    workout_id: str
    activity_type: str
    duration_minutes: int
    intensity: str
    calories_burned: int
    timestamp: datetime
    
class FitbitDailySync(BaseModel):
    date: str  # YYYY-MM-DD
    steps: int
    active_minutes: int
    calories_burned: int
    sleep_hours: float
    avg_heart_rate: int
    
class FitbitState(BaseModel):
    user_id: str
    goals: List[FitbitGoal] = Field(default_factory=list)
    workout_history: List[FitbitWorkout] = Field(default_factory=list)
    daily_syncs: List[FitbitDailySync] = Field(default_factory=list)
    
# ==================== Chase ====================
class ChaseTransaction(BaseModel):
    transaction_id: str
    date: str  # YYYY-MM-DD
    merchant: str
    amount: float
    transaction_type: str  # "debit" or "credit"
    category: str  # "dining", "shopping", "transportation", etc.
    
class ChaseAccount(BaseModel):
    account_id: str
    account_type: str  # "checking", "savings", "credit_card"
    balance: float
    
class ChaseState(BaseModel):
    user_id: str
    accounts: List[ChaseAccount] = Field(default_factory=list)
    transaction_history: List[ChaseTransaction] = Field(default_factory=list)
    
# ==================== Robinhood ====================
class RobinhoodHolding(BaseModel):
    symbol: str
    asset_type: str  # "stock" or "crypto"
    quantity: float
    average_buy_price: float
    
class RobinhoodTransaction(BaseModel):
    transaction_id: str
    symbol: str
    asset_type: str
    transaction_type: str  # "buy" or "sell"
    quantity: float
    price: float
    timestamp: datetime
    
class RobinhoodState(BaseModel):
    user_id: str
    cash_balance: float
    holdings: List[RobinhoodHolding] = Field(default_factory=list)
    watchlist: List[str] = Field(default_factory=list)  # symbols
    transaction_history: List[RobinhoodTransaction] = Field(default_factory=list)
    
# ==================== WhatsApp ====================
class WhatsAppMessage(BaseModel):
    message_id: str
    from_user: str
    to_user: str  # or group_id
    message_type: str  # "text" or "media"
    content: str
    timestamp: datetime
    
class WhatsAppState(BaseModel):
    user_id: str
    contacts: List[str] = Field(default_factory=list)
    message_history: List[WhatsAppMessage] = Field(default_factory=list)
    
# ==================== Gmail ====================
class GmailEmail(BaseModel):
    email_id: str
    from_address: str
    to_address: str
    subject: str
    body: str
    timestamp: datetime
    is_read: bool = False
    labels: List[str] = Field(default_factory=list)
    
class GmailState(BaseModel):
    user_id: str
    email_address: str
    inbox: List[GmailEmail] = Field(default_factory=list)
    sent_emails: List[GmailEmail] = Field(default_factory=list)
    
# ==================== LinkedIn ====================
class LinkedInExperience(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    
class LinkedInPost(BaseModel):
    post_id: str
    author: str
    content: str
    timestamp: datetime
    likes_count: int = 0
    
class LinkedInState(BaseModel):
    user_id: str
    headline: str = ""
    summary: str = ""
    experiences: List[LinkedInExperience] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    connections: List[str] = Field(default_factory=list)  # user_ids
    posts: List[LinkedInPost] = Field(default_factory=list)
    
# ==================== Notion ====================
class NotionPage(BaseModel):
    page_id: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    
class NotionDatabaseEntry(BaseModel):
    entry_id: str
    database_name: str  # "tasks", "habits", "projects"
    properties: Dict[str, Any]
    created_at: datetime
    
class NotionState(BaseModel):
    user_id: str
    pages: List[NotionPage] = Field(default_factory=list)
    database_entries: List[NotionDatabaseEntry] = Field(default_factory=list)
    
# ==================== Netflix ====================
class NetflixTitle(BaseModel):
    title_id: str
    name: str
    content_type: str  # "movie" or "series"
    genre: str
    
class NetflixViewHistory(BaseModel):
    title_id: str
    watched_at: datetime
    duration_watched: int  # minutes
    completed: bool
    
class NetflixState(BaseModel):
    user_id: str
    subscription_plan: str = "Standard"  # "Basic", "Standard", "Premium"
    my_list: List[str] = Field(default_factory=list)  # title_ids
    watch_history: List[NetflixViewHistory] = Field(default_factory=list)
    
# ==================== Goodreads ====================
class GoodreadsBook(BaseModel):
    book_id: str
    title: str
    author: str
    genre: str
    
class GoodreadsShelfEntry(BaseModel):
    book_id: str
    shelf: str  # "want-to-read", "currently-reading", "read"
    added_at: datetime
    
class GoodreadsReview(BaseModel):
    book_id: str
    rating: int  # 1-5
    review_text: Optional[str] = None
    reviewed_at: datetime
    
class GoodreadsState(BaseModel):
    user_id: str
    shelves: List[GoodreadsShelfEntry] = Field(default_factory=list)
    reviews: List[GoodreadsReview] = Field(default_factory=list)
    
# ==================== Instagram ====================
class InstagramPost(BaseModel):
    post_id: str
    author: str
    content_type: str  # "photo", "video", "story"
    caption: str
    timestamp: datetime
    likes_count: int = 0
    
class InstagramState(BaseModel):
    user_id: str
    followers: List[str] = Field(default_factory=list)
    following: List[str] = Field(default_factory=list)
    posts: List[InstagramPost] = Field(default_factory=list)
    
# ==================== LLM Assistant ====================
class LLMMessage(BaseModel):
    message_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    
class LLMConversation(BaseModel):
    conversation_id: str
    messages: List[LLMMessage] = Field(default_factory=list)
    created_at: datetime
    
class LLMState(BaseModel):
    user_id: str
    conversations: List[LLMConversation] = Field(default_factory=list)
    
# ==================== Google ====================
class GoogleSearchResult(BaseModel):
    result_id: str
    title: str
    snippet: str
    
class GoogleSearchHistory(BaseModel):
    query: str
    results: List[GoogleSearchResult]
    searched_at: datetime
    clicked_result_id: Optional[str] = None  # which result was clicked
    
class GoogleState(BaseModel):
    user_id: str
    search_history: List[GoogleSearchHistory] = Field(default_factory=list)



# ==================== Amazon APIs ====================
class SearchProductsInput(BaseModel):
    query: str
    
class SearchProductsOutput(BaseModel):
    products: List[AmazonProduct]
    search_timestamp: datetime
    
class ShowProductInput(BaseModel):
    product_id: str
    
class ShowProductOutput(BaseModel):
    product: AmazonProduct
    reviews: List[Dict[str, Any]]  # {rating, text, author}
    in_cart: bool
    in_wishlist: bool
    
class AddToCartInput(BaseModel):
    product_id: str
    quantity: int = 1
    
class AddToCartOutput(BaseModel):
    success: bool
    cart: List[Dict[str, Any]]  # updated cart
    cart_total: float
    
class ShowCartInput(BaseModel):
    pass  # no input needed
    
class ShowCartOutput(BaseModel):
    cart_items: List[Dict[str, Any]]  # {product_id, name, price, quantity}
    cart_total: float
    prime_member: bool
    
class ShowWishlistInput(BaseModel):
    pass
    
class ShowWishlistOutput(BaseModel):
    wishlist_items: List[Dict[str, Any]]  # {product_id, name, price, added_date}
    
class CheckoutInput(BaseModel):
    pass  # checkout all items in cart
    
class CheckoutOutput(BaseModel):
    order_id: str
    order_items: List[Dict[str, Any]]
    total_price: float
    order_date: datetime
    estimated_delivery: str  # YYYY-MM-DD
    prime_member: bool

# ==================== Spotify APIs ====================
class SearchSongsInput(BaseModel):
    query: str
    
class SearchSongsOutput(BaseModel):
    songs: List[SpotifySong]
    search_timestamp: datetime
    
class PlaySongInput(BaseModel):
    song_id: str
    
class PlaySongOutput(BaseModel):
    song: SpotifySong
    playing_status: str  # "playing"
    premium: bool  # affects quality/ads
    play_started_at: datetime
    
class AddToPlaylistInput(BaseModel):
    playlist_id: str
    song_id: str
    
class AddToPlaylistOutput(BaseModel):
    success: bool
    playlist: SpotifyPlaylist  # updated playlist
    
class FollowArtistInput(BaseModel):
    artist_id: str
    artist_name: str
    
class FollowArtistOutput(BaseModel):
    success: bool
    followed_artists: List[str]  # updated list

# ==================== Fitbit APIs ====================
class LogWorkoutInput(BaseModel):
    activity_type: str
    duration_minutes: int
    intensity: str  # "low", "medium", "high"
    
class LogWorkoutOutput(BaseModel):
    workout: FitbitWorkout
    calories_burned: int
    today_total_active_minutes: int
    
class SyncDeviceInput(BaseModel):
    device_name: str = "Fitbit Device"
    
class SyncDeviceOutput(BaseModel):
    sync_data: FitbitDailySync
    sync_timestamp: datetime
    
class SetGoalsInput(BaseModel):
    goals: List[FitbitGoal]
    
class SetGoalsOutput(BaseModel):
    goals: List[FitbitGoal]
    updated_at: datetime

# ==================== Chase APIs ====================
class GetBalanceInput(BaseModel):
    account_id: Optional[str] = None  # if None, return all accounts
    
class GetBalanceOutput(BaseModel):
    accounts: List[ChaseAccount]
    total_balance: float
    last_updated: datetime
    
class GetTransactionsInput(BaseModel):
    account_id: Optional[str] = None
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None
    limit: int = 50
    
class GetTransactionsOutput(BaseModel):
    transactions: List[ChaseTransaction]
    account_balance: float
    
class SearchTransactionsInput(BaseModel):
    query: str  # search by merchant name or category
    
class SearchTransactionsOutput(BaseModel):
    transactions: List[ChaseTransaction]
    
class TransferMoneyInput(BaseModel):
    from_account_id: str
    to_account_id: str
    amount: float
    
class TransferMoneyOutput(BaseModel):
    success: bool
    transaction_id: str
    from_account_new_balance: float
    to_account_new_balance: float
    timestamp: datetime
    
class PayBillInput(BaseModel):
    biller_name: str
    amount: float
    from_account_id: str
    
class PayBillOutput(BaseModel):
    success: bool
    transaction_id: str
    new_balance: float
    timestamp: datetime

# ==================== Robinhood APIs ====================
class GetPortfolioInput(BaseModel):
    pass
    
class GetPortfolioOutput(BaseModel):
    cash_balance: float
    holdings: List[RobinhoodHolding]
    total_portfolio_value: float
    
class GetWatchlistInput(BaseModel):
    pass
    
class GetWatchlistOutput(BaseModel):
    watchlist: List[Dict[str, Any]]  # {symbol, current_price, change_percent}
    
class SearchStocksInput(BaseModel):
    query: str
    
class SearchStocksOutput(BaseModel):
    results: List[Dict[str, Any]]  # {symbol, name, current_price}
    
class GetStockQuoteInput(BaseModel):
    symbol: str
    
class GetStockQuoteOutput(BaseModel):
    symbol: str
    current_price: float
    change_percent: float
    timestamp: datetime
    in_watchlist: bool
    
class BuyStockInput(BaseModel):
    symbol: str
    quantity: float
    asset_type: str  # "stock" or "crypto"
    
class BuyStockOutput(BaseModel):
    success: bool
    transaction: RobinhoodTransaction
    new_cash_balance: float
    new_holding: RobinhoodHolding
    
class SellStockInput(BaseModel):
    symbol: str
    quantity: float
    asset_type: str
    
class SellStockOutput(BaseModel):
    success: bool
    transaction: RobinhoodTransaction
    new_cash_balance: float
    remaining_holding: Optional[RobinhoodHolding]

# ==================== WhatsApp APIs ====================
class GetMessagesInput(BaseModel):
    contact_id: str
    limit: int = 50
    
class GetMessagesOutput(BaseModel):
    contact_id: str
    messages: List[WhatsAppMessage]
    
class SendMessageInput(BaseModel):
    to: str  # contact_id
    message: str
    
class SendMessageOutput(BaseModel):
    message: WhatsAppMessage
    sent_timestamp: datetime
    
class SendMediaInput(BaseModel):
    to: str
    media_type: str  # "photo", "video", "voice"
    caption: Optional[str] = None
    
class SendMediaOutput(BaseModel):
    message: WhatsAppMessage
    sent_timestamp: datetime

# ==================== Gmail APIs ====================
class GetInboxInput(BaseModel):
    limit: int = 50
    unread_only: bool = False
    
class GetInboxOutput(BaseModel):
    emails: List[Dict[str, Any]]  # {email_id, from, subject, snippet, timestamp, is_read}
    unread_count: int
    
class ReadEmailInput(BaseModel):
    email_id: str
    
class ReadEmailOutput(BaseModel):
    email: GmailEmail
    
class SendEmailInput(BaseModel):
    to: str
    subject: str
    body: str
    
class SendEmailOutput(BaseModel):
    email: GmailEmail
    sent_timestamp: datetime
    
class ReplyEmailInput(BaseModel):
    email_id: str  # replying to this email
    body: str
    
class ReplyEmailOutput(BaseModel):
    email: GmailEmail
    sent_timestamp: datetime

# ==================== LinkedIn APIs ====================
class UpdateProfileInput(BaseModel):
    headline: Optional[str] = None
    summary: Optional[str] = None
    
class UpdateProfileOutput(BaseModel):
    success: bool
    updated_fields: Dict[str, str]
    
class AddExperienceInput(BaseModel):
    company: str
    title: str
    start_date: str  # YYYY-MM
    end_date: Optional[str] = None
    
class AddExperienceOutput(BaseModel):
    experience: LinkedInExperience
    total_experiences: int
    
class AddSkillInput(BaseModel):
    skill: str
    
class AddSkillOutput(BaseModel):
    success: bool
    skills: List[str]  # updated skills list
    
class PostUpdateInput(BaseModel):
    content: str
    
class PostUpdateOutput(BaseModel):
    post: LinkedInPost
    
class GetFeedInput(BaseModel):
    limit: int = 20
    
class GetFeedOutput(BaseModel):
    posts: List[LinkedInPost]
    
class LikePostInput(BaseModel):
    post_id: str
    
class LikePostOutput(BaseModel):
    success: bool
    post_id: str
    new_likes_count: int
    
class CommentOnPostInput(BaseModel):
    post_id: str
    comment: str
    
class CommentOnPostOutput(BaseModel):
    success: bool
    post_id: str
    comment_timestamp: datetime
    
class SearchJobsInput(BaseModel):
    query: str
    location: Optional[str] = None
    
class SearchJobsOutput(BaseModel):
    jobs: List[Dict[str, Any]]  # {job_id, title, company, location}
    
class ApplyJobInput(BaseModel):
    job_id: str
    
class ApplyJobOutput(BaseModel):
    success: bool
    job_id: str
    applied_at: datetime
    
class SendConnectionRequestInput(BaseModel):
    user_id: str
    message: Optional[str] = None
    
class SendConnectionRequestOutput(BaseModel):
    success: bool
    user_id: str
    sent_at: datetime

# ==================== Notion APIs ====================
class GetPagesInput(BaseModel):
    limit: int = 50
    
class GetPagesOutput(BaseModel):
    pages: List[Dict[str, Any]]  # {page_id, title, created_at, updated_at}
    
class CreatePageInput(BaseModel):
    title: str
    content: str
    
class CreatePageOutput(BaseModel):
    page: NotionPage
    
class UpdatePageInput(BaseModel):
    page_id: str
    title: Optional[str] = None
    content: Optional[str] = None
    
class UpdatePageOutput(BaseModel):
    page: NotionPage
    
class SearchContentInput(BaseModel):
    query: str
    
class SearchContentOutput(BaseModel):
    results: List[Dict[str, Any]]  # {page_id, title, snippet}
    
class CreateDatabaseEntryInput(BaseModel):
    database_name: str
    properties: Dict[str, Any]
    
class CreateDatabaseEntryOutput(BaseModel):
    entry: NotionDatabaseEntry

# ==================== Netflix APIs ====================
class SearchContentInput(BaseModel):
    query: str
    
class SearchContentOutput(BaseModel):
    titles: List[NetflixTitle]
    
class ShowTitleInput(BaseModel):
    title_id: str
    
class ShowTitleOutput(BaseModel):
    title: NetflixTitle
    description: str
    rating: float
    in_my_list: bool
    
class PlayContentInput(BaseModel):
    title_id: str
    
class PlayContentOutput(BaseModel):
    title: NetflixTitle
    playing_status: str
    subscription_plan: str
    play_started_at: datetime
    
class AddToMyListInput(BaseModel):
    title_id: str
    
class AddToMyListOutput(BaseModel):
    success: bool
    my_list: List[str]  # updated list of title_ids
    
class RateContentInput(BaseModel):
    title_id: str
    rating: int  # thumbs up (1) or down (0)
    
class RateContentOutput(BaseModel):
    success: bool
    title_id: str
    rating: int

# ==================== Goodreads APIs ====================
class SearchBooksInput(BaseModel):
    query: str
    
class SearchBooksOutput(BaseModel):
    books: List[GoodreadsBook]
    
class ShowBookInput(BaseModel):
    book_id: str
    
class ShowBookOutput(BaseModel):
    book: GoodreadsBook
    description: str
    average_rating: float
    on_shelf: Optional[str] = None  # which shelf it's on, if any
    
class AddToShelfInput(BaseModel):
    book_id: str
    shelf: str  # "want-to-read", "currently-reading", "read"
    
class AddToShelfOutput(BaseModel):
    success: bool
    shelf_entry: GoodreadsShelfEntry
    
class RateBookInput(BaseModel):
    book_id: str
    rating: int  # 1-5
    
class RateBookOutput(BaseModel):
    success: bool
    rating: int
    rated_at: datetime
    
class WriteReviewInput(BaseModel):
    book_id: str
    review_text: str
    rating: Optional[int] = None
    
class WriteReviewOutput(BaseModel):
    review: GoodreadsReview

# ==================== Instagram APIs ====================
class PostStoryInput(BaseModel):
    content_type: str  # "photo" or "video"
    caption: Optional[str] = None
    
class PostStoryOutput(BaseModel):
    post: InstagramPost
    
class LikePostInput(BaseModel):
    post_id: str
    
class LikePostOutput(BaseModel):
    success: bool
    post_id: str
    new_likes_count: int
    
class CommentOnPostInput(BaseModel):
    post_id: str
    comment: str
    
class CommentOnPostOutput(BaseModel):
    success: bool
    post_id: str
    comment_timestamp: datetime
    
class SendDirectMessageInput(BaseModel):
    to_user_id: str
    message: str
    
class SendDirectMessageOutput(BaseModel):
    message_id: str
    sent_timestamp: datetime
    
class FollowUserInput(BaseModel):
    user_id: str
    
class FollowUserOutput(BaseModel):
    success: bool
    following: List[str]  # updated following list
    
class UnfollowUserInput(BaseModel):
    user_id: str
    
class UnfollowUserOutput(BaseModel):
    success: bool
    following: List[str]
    
class GetFollowingInput(BaseModel):
    pass
    
class GetFollowingOutput(BaseModel):
    following: List[str]
    following_count: int

# ==================== LLM Assistant APIs ====================
class CreateConversationInput(BaseModel):
    initial_message: Optional[str] = None
    
class CreateConversationOutput(BaseModel):
    conversation_id: str
    created_at: datetime
    
class ContinueConversationInput(BaseModel):
    conversation_id: str
    message: str
    
class ContinueConversationOutput(BaseModel):
    conversation_id: str
    user_message: LLMMessage
    assistant_response: LLMMessage

# ==================== Google APIs ====================
class GoogleSearchInput(BaseModel):
    query: str
    
class GoogleSearchOutput(BaseModel):
    results: List[GoogleSearchResult]
    search_timestamp: datetime
    
class ClickResultInput(BaseModel):
    result_id: str
    search_query: str  # to link back to which search
    
class ClickResultOutput(BaseModel):
    result: GoogleSearchResult
    clicked_at: datetime
