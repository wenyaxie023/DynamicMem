"""
Pydantic models for app state and API schemas.
"""

from __future__ import annotations

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==================== Shared Enums ====================
class UberEatsOrderStatus(str, Enum):
    PLACED = "placed"
    PREPARING = "preparing"
    ON_THE_WAY = "on_the_way"
    DELIVERED = "delivered"


class GoogleMapsTravelMode(str, Enum):
    DRIVING = "driving"
    WALKING = "walking"
    TRANSIT = "transit"
    BICYCLING = "bicycling"


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"


class TransactionType(str, Enum):
    BUY = "buy"
    SELL = "sell"


class ChaseTransactionType(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


class ChaseAccountType(str, Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT_CARD = "credit_card"


class ChaseCategory(str, Enum):
    DINING = "dining"
    SHOPPING = "shopping"
    TRANSPORTATION = "transportation"
    GROCERIES = "groceries"
    ENTERTAINMENT = "entertainment"
    UTILITIES = "utilities"
    BILLS = "bills"
    OTHER = "other"


class MessageType(str, Enum):
    TEXT = "text"
    MEDIA = "media"


class MediaType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"
    VOICE = "voice"


class ContentType(str, Enum):
    """For Netflix content type"""
    MOVIE = "movie"
    SERIES = "series"


class InstagramContentType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"
    STORY = "story"


class NetflixPlan(str, Enum):
    BASIC = "Basic"
    STANDARD = "Standard"
    PREMIUM = "Premium"


class BookShelf(str, Enum):
    WANT_TO_READ = "want-to-read"
    CURRENTLY_READING = "currently-reading"
    READ = "read"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class PlayingStatus(str, Enum):
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


class WorkoutIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FitbitGoalType(str, Enum):
    STEPS = "steps"
    ACTIVE_MINUTES = "active_minutes"
    WEIGHT = "weight"
    CALORIES = "calories"
    SLEEP_HOURS = "sleep_hours"


class ThumbsRating(int, Enum):
    DOWN = 0
    UP = 1


# ==================== Shared Sub-Models for APIs ====================
class CartItem(BaseModel):
    """Item in shopping cart"""
    product_id: str
    name: str
    price: Decimal = Field(description="Price in USD")
    quantity: int = Field(ge=1, description="Number of items")


class WishlistItem(BaseModel):
    """Item in wishlist"""
    product_id: str
    name: str
    price: Decimal = Field(description="Price in USD")
    added_at: datetime = Field(description="When the item was added to wishlist")


class OrderItem(BaseModel):
    """Item in an order"""
    product_id: str
    name: str
    price: Decimal = Field(description="Price in USD")
    quantity: int = Field(ge=1, description="Number of items")


class ProductReview(BaseModel):
    """Product review"""
    rating: int = Field(ge=1, le=5, description="Rating")
    text: str = Field(description="Review content")
    author: str = Field(description="Reviewer's display name")


class WatchlistItem(BaseModel):
    """Stock/crypto in watchlist"""
    symbol: str = Field(description="Stock ticker or crypto symbol (e.g., AAPL, BTC)")
    current_price: Decimal = Field(description="Current market price in USD")
    change_percent: Decimal = Field(description="Price change percentage from previous close")


class StockSearchResult(BaseModel):
    """Stock search result"""
    symbol: str = Field(description="Stock ticker or crypto symbol")
    name: str = Field(description="Company or cryptocurrency name")
    current_price: Decimal = Field(description="Current market price in USD")


class EmailPreview(BaseModel):
    """Email preview for inbox"""
    email_id: str
    from_address: str
    subject: str
    snippet: str = Field(max_length=50, description="Preview of email body")
    timestamp: datetime
    is_read: bool


class JobListing(BaseModel):
    """Job listing"""
    job_id: str
    title: str = Field(description="Job title/position")
    company: str
    location: str = Field(description="Job location or 'Remote'")


class PagePreview(BaseModel):
    """Notion page preview"""
    page_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class NotionSearchResult(BaseModel):
    """Notion search result"""
    page_id: str
    title: str
    snippet: str = Field(description="Matching text snippet from the page")


# ==================== Amazon ====================
class AmazonProduct(BaseModel):
    product_id: str
    name: str
    price: Decimal = Field(description="Price in USD")
    category: str
    rating: float = Field(ge=1.0, le=5.0, description="Average rating")
    
class AmazonOrder(BaseModel):
    order_id: str
    product_id: str
    product_name: str
    quantity: int = Field(ge=1)
    total_price: Decimal = Field(description="Total price in USD")
    order_date: datetime
    
class AmazonState(BaseModel):
    user_id: str
    prime_member: bool = False
    order_history: List[AmazonOrder] = Field(default_factory=list)
    search_history: List[str] = Field(default_factory=list, description="List of search queries")
    viewed_products: List[AmazonProduct] = Field(default_factory=list, description="Products the user has viewed")
    cart: List[CartItem] = Field(default_factory=list)
    wishlist: List[AmazonProduct] = Field(default_factory=list, description="Products saved to wishlist")

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
    duration_minutes: int = Field(ge=0, description="Song duration in minutes")
    
class SpotifyPlayHistory(BaseModel):
    song_id: str
    played_at: datetime
    duration_played_minutes: int = Field(ge=0, description="Minutes actually played")
    
class SpotifyState(BaseModel):
    user_id: str
    premium: bool = False
    playlists: List[SpotifyPlaylist] = Field(default_factory=list)
    followed_artists: List[str] = Field(default_factory=list, description="List of artist IDs")
    play_history: List[SpotifyPlayHistory] = Field(default_factory=list)
    favorite_genres: List[str] = Field(default_factory=list)
    
# ==================== Fitbit ====================
class FitbitGoal(BaseModel):
    goal_type: FitbitGoalType
    target_value: float = Field(description="Target value for the goal (steps count, minutes, weight in lbs, etc.)")
    
class FitbitWorkout(BaseModel):
    workout_id: str
    activity_type: str = Field(description="Type of activity (e.g., running, cycling, swimming, yoga)")
    duration_minutes: int = Field(ge=0, description="Workout duration in minutes")
    intensity: WorkoutIntensity
    calories_burned: int = Field(ge=0)
    timestamp: datetime
    location: Optional[str] = Field(default=None, description="Location where workout was performed (e.g., gym name, park, home)")
    location_address: Optional[str] = Field(default=None, description="Full address of the workout location")


class FitbitActivity(BaseModel):
    """A recorded activity with location tracking"""
    activity_id: str
    activity_type: str = Field(description="Type of activity (e.g., walking, hiking, outdoor_run)")
    duration_minutes: int = Field(ge=0)
    distance_km: Optional[float] = Field(default=None, ge=0, description="Distance covered in kilometers")
    calories_burned: int = Field(ge=0)
    timestamp: datetime
    start_location: Optional[str] = Field(default=None, description="Starting location name")
    end_location: Optional[str] = Field(default=None, description="Ending location name")
    route_description: Optional[str] = Field(default=None, description="Description of the route taken")
    
class FitbitDailySync(BaseModel):
    sync_date: date = Field(description="Date of the sync data")
    steps: int = Field(ge=0)
    active_minutes: int = Field(ge=0)
    calories_burned: int = Field(ge=0)
    sleep_hours: float = Field(ge=0, description="Hours of sleep")
    avg_heart_rate: int = Field(ge=0, description="Average heart rate in BPM")
    
class FitbitState(BaseModel):
    user_id: str
    goals: List[FitbitGoal] = Field(default_factory=list)
    workout_history: List[FitbitWorkout] = Field(default_factory=list)
    activity_history: List[FitbitActivity] = Field(default_factory=list)
    daily_syncs: List[FitbitDailySync] = Field(default_factory=list)
    
# ==================== Chase ====================
class ChaseTransaction(BaseModel):
    transaction_id: str
    transaction_date: date = Field(description="Date of the transaction")
    merchant: str
    amount: Decimal = Field(description="Transaction amount in USD")
    transaction_type: ChaseTransactionType
    category: ChaseCategory
    
class ChaseAccount(BaseModel):
    account_id: str
    account_type: ChaseAccountType
    balance: Decimal = Field(description="Current balance in USD")
    
class ChaseState(BaseModel):
    user_id: str
    accounts: List[ChaseAccount] = Field(default_factory=list)
    transaction_history: List[ChaseTransaction] = Field(default_factory=list)
    
# ==================== Robinhood ====================
class RobinhoodHolding(BaseModel):
    symbol: str = Field(description="Stock ticker or crypto symbol")
    asset_type: AssetType
    quantity: float = Field(ge=0, description="Number of shares or units held")
    average_buy_price: Decimal = Field(description="Average purchase price in USD")
    
class RobinhoodTransaction(BaseModel):
    transaction_id: str
    symbol: str
    asset_type: AssetType
    transaction_type: TransactionType
    quantity: float = Field(ge=0)
    price: Decimal = Field(description="Price per share/unit in USD")
    timestamp: datetime
    
class RobinhoodState(BaseModel):
    user_id: str
    cash_balance: Decimal = Field(description="Available cash in USD")
    holdings: List[RobinhoodHolding] = Field(default_factory=list)
    watchlist: List[str] = Field(default_factory=list, description="List of stock/crypto symbols being watched")
    transaction_history: List[RobinhoodTransaction] = Field(default_factory=list)
    
# ==================== WhatsApp ====================
class WhatsAppMessage(BaseModel):
    message_id: str
    from_user: str = Field(description="Sender's user ID or phone number")
    to_user: str = Field(description="Recipient's user ID, phone number, or group ID")
    message_type: MessageType
    content: str
    timestamp: datetime
    
class WhatsAppState(BaseModel):
    user_id: str
    contacts: List[str] = Field(default_factory=list, description="List of contact names or IDs")
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
    labels: List[str] = Field(default_factory=list, description="Email labels (e.g., inbox, sent, important)")
    
class GmailState(BaseModel):
    user_id: str
    email_address: str
    inbox: List[GmailEmail] = Field(default_factory=list)
    sent_emails: List[GmailEmail] = Field(default_factory=list)
    
# ==================== LinkedIn ====================
class LinkedInExperience(BaseModel):
    company: str
    title: str
    start_date: date = Field(description="Start date")
    end_date: Optional[date] = Field(default=None, description="End date, None if current position")
    
class LinkedInPost(BaseModel):
    post_id: str
    author: str = Field(description="Author's user ID or name")
    content: str
    timestamp: datetime
    likes_count: int = Field(default=0, ge=0)
    
class LinkedInState(BaseModel):
    user_id: str
    headline: str = Field(default="", description="Professional headline")
    summary: str = Field(default="", description="Profile summary/about section")
    experiences: List[LinkedInExperience] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    connections: List[str] = Field(default_factory=list, description="List of connected user IDs")
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
    database_name: str = Field(description="Database name (e.g., tasks, habits, projects)")
    properties: Dict[str, Any] = Field(description="Dynamic properties of the entry")
    created_at: datetime
    
class NotionState(BaseModel):
    user_id: str
    pages: List[NotionPage] = Field(default_factory=list)
    database_entries: List[NotionDatabaseEntry] = Field(default_factory=list)
    
# ==================== Netflix ====================
class NetflixTitle(BaseModel):
    title_id: str
    name: str
    content_type: ContentType
    genre: str
    
class NetflixViewHistory(BaseModel):
    title_id: str
    watched_at: datetime
    duration_watched_minutes: int = Field(ge=0, description="Duration watched in minutes")
    completed: bool = Field(description="Whether the content was watched to completion")
    
class NetflixState(BaseModel):
    user_id: str
    subscription_plan: NetflixPlan = NetflixPlan.STANDARD
    my_list: List[str] = Field(default_factory=list, description="List of title IDs in user's list")
    watch_history: List[NetflixViewHistory] = Field(default_factory=list)
    
# ==================== Goodreads ====================
class GoodreadsBook(BaseModel):
    book_id: str
    title: str
    author: str
    genre: str
    
class GoodreadsShelfEntry(BaseModel):
    book_id: str
    shelf: BookShelf
    added_at: datetime
    
class GoodreadsReview(BaseModel):
    book_id: str
    rating: int = Field(ge=1, le=5, description="Rating from 1 (worst) to 5 (best)")
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
    content_type: InstagramContentType
    caption: str
    timestamp: datetime
    likes_count: int = Field(default=0, ge=0)
    location: Optional[str] = Field(default=None, description="Location tag for the post (e.g., place name)")
    location_address: Optional[str] = Field(default=None, description="Full address of the tagged location")
    
class InstagramState(BaseModel):
    user_id: str
    followers: List[str] = Field(default_factory=list, description="List of follower user IDs")
    following: List[str] = Field(default_factory=list, description="List of followed user IDs")
    posts: List[InstagramPost] = Field(default_factory=list)
    
# ==================== LLM Assistant ====================
class LLMMessage(BaseModel):
    message_id: str
    role: ChatRole
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
    snippet: str = Field(description="Brief description of the search result")
    
class GoogleSearchHistory(BaseModel):
    query: str
    results: List[GoogleSearchResult]
    searched_at: datetime
    clicked_result_id: Optional[str] = Field(default=None, description="ID of the result that was clicked, if any")
    
class GoogleState(BaseModel):
    user_id: str
    search_history: List[GoogleSearchHistory] = Field(default_factory=list)


# ==================== Google Maps ====================
class Location(BaseModel):
    """Geographic location with coordinates and address"""
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    address: Optional[str] = Field(default=None, description="Human-readable address")
    place_name: Optional[str] = Field(default=None, description="Name of the place (e.g., restaurant, park)")
    city: Optional[str] = Field(default=None, description="City name")
    country: Optional[str] = Field(default=None, description="Country name")


class GoogleMapsPlace(BaseModel):
    """A place from Google Maps search"""
    place_id: str
    name: str
    address: str
    category: str = Field(description="Category like restaurant, gym, park, etc.")
    rating: Optional[float] = Field(default=None, ge=0, le=5, description="Average rating")
    location: Location


class GoogleMapsCheckIn(BaseModel):
    """A check-in at a location"""
    checkin_id: str
    location: Location
    timestamp: datetime
    note: Optional[str] = None


class GoogleMapsSharedLocation(BaseModel):
    """Shared location with another user"""
    share_id: str
    shared_with: str = Field(description="User ID or name of the person shared with")
    location: Location
    shared_at: datetime
    expires_at: Optional[datetime] = None


class GoogleMapsDirections(BaseModel):
    """Directions between two locations"""
    directions_id: str
    origin: Location
    destination: Location
    travel_mode: GoogleMapsTravelMode = Field(description="Mode of travel: driving, walking, transit, bicycling")
    distance_km: float
    duration_minutes: int
    searched_at: datetime


class GoogleMapsState(BaseModel):
    user_id: str
    home_location: Optional[Location] = None
    work_location: Optional[Location] = None
    saved_places: List[GoogleMapsPlace] = Field(default_factory=list)
    checkin_history: List[GoogleMapsCheckIn] = Field(default_factory=list)
    shared_locations: List[GoogleMapsSharedLocation] = Field(default_factory=list)
    directions_history: List[GoogleMapsDirections] = Field(default_factory=list)


# ==================== UberEats ====================
class UberEatsRestaurant(BaseModel):
    """A restaurant on UberEats"""
    restaurant_id: str
    name: str
    cuisine_type: str = Field(description="Type of cuisine (e.g., Italian, Chinese, Mexican)")
    rating: float = Field(ge=0, le=5, description="Average rating")
    delivery_time_minutes: int = Field(description="Estimated delivery time in minutes")
    delivery_fee: Decimal = Field(description="Delivery fee in USD")
    address: str
    location: Optional[Location] = None


class UberEatsMenuItem(BaseModel):
    """An item from a restaurant menu"""
    item_id: str
    name: str
    description: str
    price: Decimal = Field(description="Price in USD")
    category: str = Field(description="Menu category like appetizers, mains, desserts")


class UberEatsOrderItem(BaseModel):
    """An item in an order"""
    item_id: str
    name: str
    quantity: int = Field(ge=1)
    price: Decimal
    special_instructions: Optional[str] = None


class UberEatsOrder(BaseModel):
    """A food delivery order"""
    order_id: str
    restaurant_id: str
    restaurant_name: str
    items: List[UberEatsOrderItem]
    subtotal: Decimal
    delivery_fee: Decimal
    tip: Decimal
    total: Decimal
    delivery_address: str
    delivery_location: Optional[Location] = None
    order_time: datetime
    estimated_delivery: datetime
    status: UberEatsOrderStatus = Field(description="Order status: placed, preparing, on_the_way, delivered")


class UberEatsState(BaseModel):
    user_id: str
    delivery_addresses: List[str] = Field(default_factory=list)
    default_address: Optional[str] = None
    default_location: Optional[Location] = None
    favorite_restaurants: List[str] = Field(default_factory=list, description="List of restaurant IDs")
    order_history: List[UberEatsOrder] = Field(default_factory=list)



# ==================== Amazon APIs ====================
class SearchProductsInput(BaseModel):
    query: str = Field(description="Search query string")
    
class SearchProductsOutput(BaseModel):
    products: List[AmazonProduct]
    search_timestamp: datetime
    
class ShowProductInput(BaseModel):
    product_id: str
    
class ShowProductOutput(BaseModel):
    product: AmazonProduct
    reviews: List[ProductReview]
    in_cart: bool
    in_wishlist: bool
    
class AddToCartInput(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=1)
    
class AddToCartOutput(BaseModel):
    cart: List[CartItem]
    cart_total: Decimal = Field(description="Total cart value in USD")
    
class ShowCartInput(BaseModel):
    pass
    
class ShowCartOutput(BaseModel):
    cart_items: List[CartItem]
    cart_total: Decimal = Field(description="Total cart value in USD")
    prime_member: bool
    
class ShowWishlistInput(BaseModel):
    pass
    
class ShowWishlistOutput(BaseModel):
    wishlist_items: List[WishlistItem]
    
class CheckoutInput(BaseModel):
    pass
    
class CheckoutOutput(BaseModel):
    order_id: str
    order_items: List[OrderItem]
    total_price: Decimal = Field(description="Total order price in USD")
    order_date: datetime
    estimated_delivery: date = Field(description="Estimated delivery date")
    prime_member: bool

# ==================== Spotify APIs ====================
class SearchSongsInput(BaseModel):
    query: str = Field(description="Search query for songs, artists, or albums")
    
class SearchSongsOutput(BaseModel):
    songs: List[SpotifySong]
    search_timestamp: datetime
    
class PlaySongInput(BaseModel):
    song_id: str
    
class PlaySongOutput(BaseModel):
    song: SpotifySong
    playing_status: PlayingStatus
    premium: bool = Field(description="Whether user has premium (affects quality/ads)")
    play_started_at: datetime
    
class AddToPlaylistInput(BaseModel):
    playlist_id: str
    song_id: str
    
class AddToPlaylistOutput(BaseModel):
    playlist: SpotifyPlaylist
    
class FollowArtistInput(BaseModel):
    artist_id: str
    artist_name: str
    
class FollowArtistOutput(BaseModel):
    followed_artists: List[str] = Field(description="Updated list of followed artist IDs")

# ==================== Fitbit APIs ====================
class LogWorkoutInput(BaseModel):
    activity_type: str = Field(description="Type of activity (e.g., running, cycling, swimming)")
    duration_minutes: int = Field(ge=1, description="Workout duration in minutes")
    intensity: WorkoutIntensity
    location: Optional[str] = Field(default=None, description="Location where workout was performed (e.g., gym name, park, home)")
    location_address: Optional[str] = Field(default=None, description="Full address of the workout location")
    
class LogWorkoutOutput(BaseModel):
    workout: FitbitWorkout
    calories_burned: int
    today_total_active_minutes: int
    
class SyncDeviceInput(BaseModel):
    """No input required - syncs the connected Fitbit device."""
    pass
    
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
    account_id: Optional[str] = Field(default=None, description="Specific account ID, or None for all accounts")
    
class GetBalanceOutput(BaseModel):
    accounts: List[ChaseAccount]
    total_balance: Decimal = Field(description="Total balance across all accounts in USD")
    last_updated: datetime
    
class GetTransactionsInput(BaseModel):
    account_id: Optional[str] = None
    start_date: Optional[date] = Field(default=None, description="Filter transactions from this date")
    end_date: Optional[date] = Field(default=None, description="Filter transactions until this date")
    
class GetTransactionsOutput(BaseModel):
    transactions: List[ChaseTransaction]
    account_balance: Decimal
    
class SearchTransactionsInput(BaseModel):
    query: str = Field(description="Search by merchant name or category")
    
class SearchTransactionsOutput(BaseModel):
    transactions: List[ChaseTransaction]
    
class TransferMoneyInput(BaseModel):
    from_account_id: str
    to_account_id: str
    amount: Decimal = Field(gt=0, description="Amount to transfer in USD")
    
class TransferMoneyOutput(BaseModel):
    transaction_id: str
    from_account_new_balance: Decimal
    to_account_new_balance: Decimal
    timestamp: datetime
    
class PayBillInput(BaseModel):
    biller_name: str
    amount: Decimal = Field(gt=0, description="Bill amount in USD")
    from_account_id: str
    
class PayBillOutput(BaseModel):
    transaction_id: str
    new_balance: Decimal = Field(description="New account balance after payment")
    timestamp: datetime

# ==================== Robinhood APIs ====================
class GetPortfolioInput(BaseModel):
    pass
    
class GetPortfolioOutput(BaseModel):
    cash_balance: Decimal
    holdings: List[RobinhoodHolding]
    total_portfolio_value: Decimal = Field(description="Total value of holdings + cash in USD")
    
class GetWatchlistInput(BaseModel):
    pass
    
class GetWatchlistOutput(BaseModel):
    watchlist: List[WatchlistItem]
    
class SearchStocksInput(BaseModel):
    query: str = Field(description="Search query for stock ticker or company name")
    
class SearchStocksOutput(BaseModel):
    results: List[StockSearchResult]
    
class GetStockQuoteInput(BaseModel):
    symbol: str = Field(description="Stock ticker or crypto symbol")
    
class GetStockQuoteOutput(BaseModel):
    symbol: str
    current_price: Decimal
    change_percent: Decimal = Field(description="Price change percentage from previous close")
    timestamp: datetime
    in_watchlist: bool
    
class BuyStockInput(BaseModel):
    symbol: str
    quantity: Decimal = Field(gt=0)
    asset_type: AssetType
    
class BuyStockOutput(BaseModel):
    transaction: RobinhoodTransaction
    new_cash_balance: Decimal
    new_holding: RobinhoodHolding
    
class SellStockInput(BaseModel):
    symbol: str
    quantity: Decimal = Field(gt=0)
    asset_type: AssetType
    
class SellStockOutput(BaseModel):
    transaction: RobinhoodTransaction
    new_cash_balance: Decimal
    remaining_holding: Optional[RobinhoodHolding] = Field(description="Remaining holding after sale, None if fully sold")

# ==================== WhatsApp APIs ====================
class GetMessagesInput(BaseModel):
    contact_id: str = Field(description="Contact or group ID to get messages for")
    
class GetMessagesOutput(BaseModel):
    contact_id: str
    messages: List[WhatsAppMessage]
    
class SendMessageInput(BaseModel):
    to: str = Field(description="Recipient's contact ID or phone number")
    message: str
    
class SendMessageOutput(BaseModel):
    message: WhatsAppMessage
    sent_timestamp: datetime
    
class SendMediaInput(BaseModel):
    to: str = Field(description="Recipient's contact ID or phone number")
    media_type: MediaType
    caption: Optional[str] = None
    
class SendMediaOutput(BaseModel):
    message: WhatsAppMessage
    sent_timestamp: datetime

# ==================== Gmail APIs ====================
class GetInboxInput(BaseModel):
    unread_only: bool = False
    
class GetInboxOutput(BaseModel):
    emails: List[EmailPreview]
    unread_count: int
    
class ReadEmailInput(BaseModel):
    email_id: str
    
class ReadEmailOutput(BaseModel):
    email: GmailEmail
    
class SendEmailInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str
    body: str
    
class SendEmailOutput(BaseModel):
    email: GmailEmail
    sent_timestamp: datetime
    
class ReplyEmailInput(BaseModel):
    email_id: str = Field(description="ID of the email being replied to")
    body: str
    
class ReplyEmailOutput(BaseModel):
    email: GmailEmail
    sent_timestamp: datetime

# ==================== LinkedIn APIs ====================
class UpdateProfileInput(BaseModel):
    headline: Optional[str] = None
    summary: Optional[str] = None
    
class UpdateProfileOutput(BaseModel):
    updated_fields: Dict[str, str] = Field(description="Map of field name to new value")
    
class AddExperienceInput(BaseModel):
    company: str
    title: str
    start_date: date = Field(description="Start date")
    end_date: Optional[date] = Field(default=None, description="End date, None if current position")
    
class AddExperienceOutput(BaseModel):
    experience: LinkedInExperience
    total_experiences: int
    
class AddSkillInput(BaseModel):
    skill: str
    
class AddSkillOutput(BaseModel):
    skills: List[str] = Field(description="Updated list of skills")
    
class PostUpdateInput(BaseModel):
    content: str = Field(description="Post content")
    
class PostUpdateOutput(BaseModel):
    post: LinkedInPost
    
class GetFeedInput(BaseModel):
    pass
    
class GetFeedOutput(BaseModel):
    posts: List[LinkedInPost]
    
class LikePostInput(BaseModel):
    post_id: str
    
class LikePostOutput(BaseModel):
    post_id: str
    new_likes_count: int
    
class CommentOnPostInput(BaseModel):
    post_id: str
    comment: str
    
class CommentOnPostOutput(BaseModel):
    post_id: str
    comment_timestamp: datetime
    
class SearchJobsInput(BaseModel):
    query: str = Field(description="Job title or keywords to search")
    location: Optional[str] = Field(default=None, description="Location filter")
    
class SearchJobsOutput(BaseModel):
    jobs: List[JobListing]
    
class ApplyJobInput(BaseModel):
    job_id: str
    
class ApplyJobOutput(BaseModel):
    job_id: str
    applied_at: datetime
    
class SendConnectionRequestInput(BaseModel):
    user_id: str = Field(description="Target user's ID")
    message: Optional[str] = Field(default=None, description="Optional connection request message")
    
class SendConnectionRequestOutput(BaseModel):
    user_id: str
    sent_at: datetime

# ==================== Notion APIs ====================
class GetPagesInput(BaseModel):
    pass
    
class GetPagesOutput(BaseModel):
    pages: List[PagePreview]
    
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
    
class NotionSearchContentInput(BaseModel):
    """Search content in Notion"""
    query: str
    
class NotionSearchContentOutput(BaseModel):
    """Search results from Notion"""
    results: List[NotionSearchResult]
    
class CreateDatabaseEntryInput(BaseModel):
    database_name: str = Field(description="Database name (e.g., tasks, habits, projects)")
    properties: Dict[str, Any] = Field(description="Entry properties as key-value pairs")
    
class CreateDatabaseEntryOutput(BaseModel):
    entry: NotionDatabaseEntry

# ==================== Netflix APIs ====================
class NetflixSearchContentInput(BaseModel):
    """Search content on Netflix"""
    query: str = Field(description="Search query for movies or series")
    
class NetflixSearchContentOutput(BaseModel):
    """Search results from Netflix"""
    titles: List[NetflixTitle]
    
class ShowTitleInput(BaseModel):
    title_id: str
    
class ShowTitleOutput(BaseModel):
    title: NetflixTitle
    description: str
    rating: float = Field(ge=0, le=5, description="Average user rating")
    in_my_list: bool
    
class PlayContentInput(BaseModel):
    title_id: str
    
class PlayContentOutput(BaseModel):
    title: NetflixTitle
    playing_status: PlayingStatus
    subscription_plan: NetflixPlan
    play_started_at: datetime
    
class AddToMyListInput(BaseModel):
    title_id: str
    
class AddToMyListOutput(BaseModel):
    my_list: List[str] = Field(description="Updated list of title IDs")
    
class RateContentInput(BaseModel):
    title_id: str
    rating: ThumbsRating
    
class RateContentOutput(BaseModel):
    title_id: str
    rating: ThumbsRating

# ==================== Goodreads APIs ====================
class SearchBooksInput(BaseModel):
    query: str = Field(description="Search query for book title or author")
    
class SearchBooksOutput(BaseModel):
    books: List[GoodreadsBook]
    
class ShowBookInput(BaseModel):
    book_id: str
    
class ShowBookOutput(BaseModel):
    book: GoodreadsBook
    description: str
    average_rating: float = Field(ge=1.0, le=5.0, description="Average rating")
    on_shelf: Optional[BookShelf] = Field(default=None, description="Which shelf the book is on, if any")
    
class AddToShelfInput(BaseModel):
    book_id: str
    shelf: BookShelf
    
class AddToShelfOutput(BaseModel):
    shelf_entry: GoodreadsShelfEntry
    
class RateBookInput(BaseModel):
    book_id: str
    rating: int = Field(ge=1, le=5, description="Rating")
    
class RateBookOutput(BaseModel):
    rating: int = Field(ge=1, le=5, description="Rating")
    rated_at: datetime
    
class WriteReviewInput(BaseModel):
    book_id: str
    review_text: str
    rating: Optional[int] = Field(default=None, ge=1, le=5, description="Optional rating")
    
class WriteReviewOutput(BaseModel):
    review: GoodreadsReview

# ==================== Instagram APIs ====================
class PostStoryInput(BaseModel):
    content_type: MediaType = Field(description="Type of story content (photo or video, voice not applicable)")
    caption: Optional[str] = None
    location: Optional[str] = Field(default=None, description="Location tag for the story")
    location_address: Optional[str] = Field(default=None, description="Full address of the tagged location")
    
class PostStoryOutput(BaseModel):
    post: InstagramPost
    
class InstagramLikePostInput(BaseModel):
    post_id: str
    
class InstagramLikePostOutput(BaseModel):
    post_id: str
    new_likes_count: int
    
class InstagramCommentOnPostInput(BaseModel):
    post_id: str
    comment: str
    
class InstagramCommentOnPostOutput(BaseModel):
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
    following: List[str] = Field(description="Updated list of followed user IDs")
    
class UnfollowUserInput(BaseModel):
    user_id: str
    
class UnfollowUserOutput(BaseModel):
    following: List[str] = Field(description="Updated list of followed user IDs")
    
class GetFollowingInput(BaseModel):
    pass
    
class GetFollowingOutput(BaseModel):
    following: List[str]
    following_count: int

# ==================== LLM Assistant APIs ====================
class CreateConversationInput(BaseModel):
    initial_message: Optional[str] = Field(default=None, description="Optional initial message to start the conversation")
    
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
    query: str = Field(description="Search query")
    
class GoogleSearchOutput(BaseModel):
    results: List[GoogleSearchResult]
    search_timestamp: datetime
    
class ClickResultInput(BaseModel):
    result_id: str
    search_query: str = Field(description="The search query this result came from")
    
class ClickResultOutput(BaseModel):
    result: GoogleSearchResult
    clicked_at: datetime


# ==================== Google Maps APIs ====================
class ShareLocationInput(BaseModel):
    shared_with: str = Field(description="User ID or name of the person to share location with")
    duration_minutes: Optional[int] = Field(default=None, description="How long to share location (None for indefinite)")
    
class ShareLocationOutput(BaseModel):
    share_id: str
    shared_location: GoogleMapsSharedLocation
    
class CheckInInput(BaseModel):
    place_name: str = Field(description="Name of the place to check in at")
    address: Optional[str] = Field(default=None, description="Address of the location")
    note: Optional[str] = Field(default=None, description="Optional note about the check-in")
    
class CheckInOutput(BaseModel):
    checkin: GoogleMapsCheckIn
    
class GetDirectionsInput(BaseModel):
    origin: str = Field(description="Starting location (address or place name)")
    destination: str = Field(description="Destination location (address or place name)")
    travel_mode: str = Field(default="driving", description="Mode: driving, walking, transit, bicycling")
    
class GetDirectionsOutput(BaseModel):
    directions: GoogleMapsDirections
    
class SearchPlacesInput(BaseModel):
    query: str = Field(description="Search query for places (e.g., 'coffee shop near me')")
    category: Optional[str] = Field(default=None, description="Category filter like restaurant, gym, park")
    
class SearchPlacesOutput(BaseModel):
    places: List[GoogleMapsPlace]
    search_timestamp: datetime


# ==================== Fitbit RecordActivity API ====================
class RecordActivityInput(BaseModel):
    activity_type: str = Field(description="Type of activity (e.g., walking, hiking, outdoor_run)")
    duration_minutes: int = Field(ge=1, description="Activity duration in minutes")
    distance_km: Optional[float] = Field(default=None, ge=0, description="Distance covered in kilometers")
    start_location: Optional[str] = Field(default=None, description="Starting location name")
    end_location: Optional[str] = Field(default=None, description="Ending location name")
    route_description: Optional[str] = Field(default=None, description="Description of the route taken")
    
class RecordActivityOutput(BaseModel):
    activity: FitbitActivity
    calories_burned: int
    today_total_active_minutes: int


# ==================== Instagram CreatePost API ====================
class CreatePostInput(BaseModel):
    content_type: InstagramContentType = Field(description="Type of post content: photo, video")
    caption: str = Field(description="Post caption text")
    location: Optional[str] = Field(default=None, description="Location tag for the post")
    location_address: Optional[str] = Field(default=None, description="Full address of the tagged location")
    
class CreatePostOutput(BaseModel):
    post: InstagramPost


# ==================== UberEats APIs ====================
class SearchRestaurantsInput(BaseModel):
    query: Optional[str] = Field(default=None, description="Search query for restaurant name or cuisine type")
    cuisine_type: Optional[str] = Field(default=None, description="Filter by cuisine type")
    
class SearchRestaurantsOutput(BaseModel):
    restaurants: List[UberEatsRestaurant]
    
class GetMenuInput(BaseModel):
    restaurant_id: str
    
class GetMenuOutput(BaseModel):
    restaurant: UberEatsRestaurant
    menu_items: List[UberEatsMenuItem]
    
class PlaceOrderInput(BaseModel):
    restaurant_id: str
    items: List[Dict[str, Any]] = Field(description="List of items with item_id, quantity, and optional special_instructions")
    delivery_address: str
    tip_amount: Optional[Decimal] = Field(default=Decimal('0.0'), ge=0, description="Tip amount in USD")
    
class PlaceOrderOutput(BaseModel):
    order: UberEatsOrder
    
class GetOrderHistoryInput(BaseModel):
    limit: int = Field(default=10, ge=1, le=50, description="Number of recent orders to retrieve")
    
class GetOrderHistoryOutput(BaseModel):
    orders: List[UberEatsOrder]
