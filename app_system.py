"""
App System for generating realistic app logs with stateful consistency.

Each app maintains state about the user to ensure consistency across:
- Different domains calling the same app
- Multiple calls within the same domain
"""

from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from mem_bench.behavior_and_conversation.app_data_sources import (
    SpotifyDatabase,
    get_data_generator
)


from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

#######新增定义 begin

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


from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

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

#######新增定义 end



@dataclass
class AppLogEntry:
    """A single app log entry."""
    timestamp: str  # "YYYY-MM-DD HH:MM:SS"
    app_name: str
    api_name: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


APP_API_SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Amazon": {
        "SearchProducts": {
            "input": {"query": "string"},
            "output": {
                "products": [
                    {"name": "string", "price": "number", "rating": "number", "description": "string"}
                ],
                "total_results": "integer"
            }
        },
        "ShowProduct": {
            "input": {"product_name": "string"},
            "output": {
                "name": "string",
                "price": "number",
                "rating": "number",
                "reviews": "integer",
                "description": "string",
                "in_stock": "boolean"
            }
        },
        "ReadReviews": {
            "input": {"product_name": "string"},
            "output": {
                "product_name": "string",
                "reviews": ["object"],
                "average_rating": "number",
                "total_reviews": "integer"
            }
        },
        "AddToCart": {
            "input": {"product_name": "string", "quantity": "integer"},
            "output": {
                "product_name": "string",
                "quantity": "integer",
                "cart_total": "number",
                "items_in_cart": "integer"
            }
        },
        "Checkout": {
            "input": {},
            "output": {
                "order_number": "string",
                "items": ["object"],
                "total_price": "number",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "estimated_delivery": "YYYY-MM-DD"
            }
        },
        "ShowOrders": {
            "input": {},
            "output": {"orders": ["object"], "total_orders": "integer"}
        },
        "TrackOrder": {
            "input": {"order_number": "string"},
            "output": {
                "order_number": "string",
                "status": "string",
                "current_location": "string",
                "estimated_delivery": "YYYY-MM-DD",
                "tracking_events": ["object"]
            }
        }
    },
    "Spotify": {
        "SearchSongs": {
            "input": {"query": "string"},
            "output": {"songs": ["object"], "total_results": "integer"}
        },
        "ShowArtist": {
            "input": {"artist_name": "string"},
            "output": {"artist": "string", "genres": ["string"], "top_songs": ["object"]}
        },
        "PlaySong": {
            "input": {"song_title": "string", "artist": "string"},
            "output": {
                "song": "object",
                "status": "string",
                "duration_seconds": "integer"
            }
        },
        "CreatePlaylist": {
            "input": {"playlist_name": "string"},
            "output": {"playlist_name": "string", "song_count": "integer", "created_at": "YYYY-MM-DD"}
        },
        "AddToPlaylist": {
            "input": {"playlist_name": "string", "song_title": "string", "artist": "string"},
            "output": {"playlist_name": "string", "song_added": "object", "song_count": "integer"}
        },
        "ShowPlaylists": {
            "input": {},
            "output": {"playlists": ["object"]}
        },
        "ShowRecentlyPlayed": {
            "input": {},
            "output": {"songs": ["object"]}
        }
    },
    "SimpleNote": {
        "ShowNotes": {
            "input": {},
            "output": {
                "notes": [{"title": "string", "preview": "string", "tags": ["string"], "created_at": "string"}],
                "total_notes": "integer"
            }
        },
        "ShowNote": {
            "input": {"note_title": "string"},
            "output": {
                "title": "string",
                "content": "string",
                "tags": ["string"],
                "created_at": "string",
                "updated_at": "string"
            }
        },
        "CreateNote": {
            "input": {"title": "string", "content": "string"},
            "output": {"title": "string", "content": "string", "tags": ["string"], "created_at": "string"}
        },
        "EditNote": {
            "input": {"note_title": "string", "content": "string"},
            "output": {"title": "string", "content": "string", "tags": ["string"], "updated_at": "string"}
        },
        "SearchNotes": {
            "input": {"query": "string"},
            "output": {"notes": ["object"], "total_results": "integer"}
        },
        "TagNote": {
            "input": {"note_title": "string", "tags": ["string"]},
            "output": {"title": "string", "tags": ["string"], "updated_at": "string"}
        }
    },
    "LLM": {
        "Chat": {
            "input": {
                "message": "string (description: The user's message to the AI assistant)"
            },
            "output": {
                "conversation": [
                    {
                        "role": "string (description: Either 'user' or 'assistant')",
                        "content": "string (description: The message content)"
                    }
                ],
                "description": "string (description: Brief summary of the conversation - what the user asked and what the assistant provided)"
            }
        }
    },
    "Google": {
        "Search": {
            "input": {"query": "string"},
            "output": {
                "results": [{"title": "string", "snippet": "string", "source": "string"}],
                "total_results": "integer"
            }
        },
        "SearchNews": {
            "input": {"query": "string"},
            "output": {
                "results": [{"title": "string", "snippet": "string", "source": "string", "date": "string"}],
                "total_results": "integer"
            }
        }
    },
    "Fitbit": {
        "LogWorkout": {
            "input": {
                "type": "string",
                "duration_minutes": "integer",
                "intensity": "string",
                "session_id": "string"
            },
            "output": {
                "workout_id": "string",
                "type": "string",
                "duration_minutes": "integer",
                "intensity": "string",
                "calories_burned": "integer",
                "heart_rate_avg": "integer",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "date": "YYYY-MM-DD"
            }
        },
        "LogActivity": {
            "input": {"steps": "integer", "calories": "integer", "active_minutes": "integer"},
            "output": {"date": "YYYY-MM-DD", "steps": "integer", "calories": "integer", "active_minutes": "integer"}
        },
        "ShowDailyStats": {
            "input": {},
            "output": {
                "date": "YYYY-MM-DD",
                "steps": "integer",
                "active_minutes": "integer",
                "calories": "integer",
                "workouts": ["string"]
            }
        },
        "ShowWeeklyStats": {
            "input": {},
            "output": {
                "period": "string",
                "total_workouts": "integer",
                "total_active_minutes": "integer",
                "avg_daily_steps": "integer"
            }
        },
        "SetGoal": {
            "input": {"goal_type": "string", "target": "integer"},
            "output": {"goals": "object"}
        },
        "ShowProgress": {
            "input": {},
            "output": {"progress": "object"}
        }
    },
    "Calendar": {
        "CreateEvent": {
            "input": {"title": "string", "date": "YYYY-MM-DD", "start_time": "HH:MM", "end_time": "HH:MM"},
            "output": {
                "event_id": "string",
                "title": "string",
                "date": "YYYY-MM-DD",
                "start_time": "HH:MM",
                "end_time": "HH:MM",
                "location": "string",
                "reminder": "string"
            }
        },
        "ShowEvents": {
            "input": {},
            "output": {"events": ["object"], "total_events": "integer"}
        },
        "EditEvent": {
            "input": {"event_id": "string", "updates": "object"},
            "output": {"event": "object"}
        },
        "SetReminder": {
            "input": {"event_id": "string", "reminder_minutes": "integer"},
            "output": {"event_id": "string", "reminder_minutes": "integer"}
        }
    },
    "Message": {
        "SendMessage": {
            "input": {"to": "string", "text": "string"},
            "output": {
                "message_id": "string",
                "from": "string",
                "to": "string",
                "text": "string",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "status": "string"
            }
        },
        "GetMessages": {
            "input": {"contact_name": "string"},
            "output": {"contact_name": "string", "messages": ["object"], "total_count": "integer"}
        },
        "SearchMessages": {
            "input": {"query": "string"},
            "output": {"messages": ["object"], "total_count": "integer"}
        },
        "CreateGroup": {
            "input": {"group_name": "string", "members": ["string"]},
            "output": {"group_id": "string", "group_name": "string", "members": ["string"]}
        },
        "ReactToMessage": {
            "input": {"message_id": "string", "reaction": "string"},
            "output": {"message_id": "string", "reaction": "string", "status": "string"}
        }
    },
    "Finance": {
        "ShowAccounts": {
            "input": {},
            "output": {"accounts": ["object"]}
        },
        "ShowTransactions": {
            "input": {},
            "output": {"transactions": ["object"], "total_transactions": "integer"}
        },
        "CreateBudget": {
            "input": {"categories": ["object"]},
            "output": {"budgets": ["object"]}
        },
        "LogExpense": {
            "input": {"amount": "number", "category": "string", "merchant": "string"},
            "output": {"transaction": "object"}
        },
        "ShowBudgetProgress": {
            "input": {},
            "output": {"categories": ["object"]}
        },
        "SetFinancialGoal": {
            "input": {"goal_name": "string", "target_amount": "number"},
            "output": {"goal": "object"}
        },
        "ShowInvestments": {
            "input": {},
            "output": {"investments": ["object"]}
        },
        "TransferMoney": {
            "input": {"from_account": "string", "to_account": "string", "amount": "number"},
            "output": {"transfer": "object"}
        }
    }
}


class BaseApp(ABC):
    """
    Base class for all apps.

    Each app maintains user-specific state to ensure consistency.
    State is initialized lazily on first API call.
    """

    def __init__(self, app_name: str, user_id: str):
        self.app_name = app_name
        self.user_id = user_id
        self.state: Dict[str, Any] = {}
        self.initialized = False

    @abstractmethod
    def _initialize_state(self) -> None:
        """Initialize app-specific state for this user."""
        pass

    def ensure_initialized(self) -> None:
        """Ensure state is initialized before API calls."""
        if not self.initialized:
            self._initialize_state()
            self.initialized = True

    @abstractmethod
    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """
        Call an API and return a log entry.

        Args:
            api_name: Name of the API to call
            timestamp: When this call happened
            description: LLM-generated description of what user is doing
            context: Additional context (user profile, domain info, etc.)
        """
        pass


class AmazonApp(BaseApp):
    """Amazon shopping app."""

    def __init__(self, user_id: str):
        super().__init__("Amazon", user_id)

    def _initialize_state(self) -> None:
        """Initialize Amazon state."""
        self.state = {
            "order_history": [],
            "search_history": [],
            "viewed_products": [],
            "cart": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Amazon API."""
        self.ensure_initialized()

        if api_name == "SearchProducts":
            return self._search_products(timestamp, description, context)
        elif api_name == "ShowProduct":
            return self._show_product(timestamp, description, context)
        elif api_name == "ReadReviews":
            return self._read_reviews(timestamp, description, context)
        elif api_name == "AddToCart":
            return self._add_to_cart(timestamp, description, context)
        elif api_name == "Checkout":
            return self._checkout(timestamp, description, context)
        elif api_name == "ShowOrders":
            return self._show_orders(timestamp, description, context)
        elif api_name == "TrackOrder":
            return self._track_order(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Amazon API: {api_name}")

    def _search_products(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle product search."""
        query = self._extract_search_query(description)
        products = self._generate_search_results(query, context)

        # Update search history
        self.state["search_history"].append({
            "timestamp": timestamp,
            "query": query,
            "results_count": len(products)
        })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchProducts",
            request={"query": query},
            response={
                "products": products,
                "total_results": len(products)
            }
        )

    def _show_product(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show product details."""
        product_name = self._extract_product_name(description)
        product = self._get_or_create_product(product_name, context)

        # Add to viewed products
        self.state["viewed_products"].append({
            "timestamp": timestamp,
            "product_name": product["name"]
        })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowProduct",
            request={"product_name": product_name},
            response=product
        )

    def _read_reviews(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Read reviews for a product."""
        product_name = self._extract_product_name(description)
        product = self._get_or_create_product(product_name, context)
        reviews = self._generate_reviews(product_name, product["rating"])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ReadReviews",
            request={"product_name": product_name},
            response={
                "product_name": product_name,
                "reviews": reviews,
                "average_rating": product["rating"],
                "total_reviews": product["reviews"]
            }
        )

    def _add_to_cart(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Add product to cart."""
        product_name = self._extract_product_name(description)
        quantity = self._extract_quantity(description)
        product = self._get_or_create_product(product_name, context)

        for item in self.state["cart"]:
            if item["product_name"].lower() == product_name.lower():
                item["quantity"] += quantity
                break
        else:
            self.state["cart"].append({
                "product_name": product["name"],
                "price": product["price"],
                "quantity": quantity
            })

        cart_total = sum(item["price"] * item["quantity"] for item in self.state["cart"])
        items_in_cart = sum(item["quantity"] for item in self.state["cart"])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="AddToCart",
            request={"product_name": product_name, "quantity": quantity},
            response={
                "product_name": product["name"],
                "quantity": quantity,
                "cart_total": round(cart_total, 2),
                "items_in_cart": items_in_cart
            }
        )

    def _checkout(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Complete purchase checkout."""
        if self.state["cart"]:
            items = [
                {
                    "product_name": item["product_name"],
                    "quantity": item["quantity"],
                    "price": item["price"]
                }
                for item in self.state["cart"]
            ]
        else:
            product_name = self._extract_product_name(description)
            quantity = self._extract_quantity(description)
            product = self._get_or_create_product(product_name, context)
            items = [{
                "product_name": product["name"],
                "quantity": quantity,
                "price": product["price"]
            }]

        total_price = sum(item["price"] * item["quantity"] for item in items)
        order_number = f"AMZ{random.randint(100000000, 999999999)}"
        order = {
            "order_number": order_number,
            "items": items,
            "total_price": round(total_price, 2),
            "timestamp": timestamp,
            "estimated_delivery": self._calculate_delivery_date(timestamp),
            "status": "processing",
            "current_location": "Fulfillment Center"
        }

        self.state["order_history"].append(order)
        self.state["cart"] = []

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Checkout",
            request={},
            response={
                "order_number": order_number,
                "items": items,
                "total_price": round(total_price, 2),
                "timestamp": timestamp,
                "estimated_delivery": order["estimated_delivery"]
            }
        )

    def _show_orders(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show order history."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowOrders",
            request={},
            response={
                "orders": self.state["order_history"][-10:],
                "total_orders": len(self.state["order_history"])
            }
        )

    def _track_order(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Track a specific order."""
        order_number = self._extract_order_number(description)
        order = None
        for existing in reversed(self.state["order_history"]):
            if existing["order_number"] == order_number:
                order = existing
                break

        if not order and self.state["order_history"]:
            order = self.state["order_history"][-1]
            order_number = order["order_number"]

        if not order:
            order_number = order_number or f"AMZ{random.randint(100000000, 999999999)}"
            order = {
                "order_number": order_number,
                "estimated_delivery": self._calculate_delivery_date(timestamp)
            }

        status = random.choice(["processing", "shipped", "out_for_delivery", "delivered"])
        current_location = random.choice(["Fulfillment Center", "Regional Hub", "Local Facility", "Out for delivery"])
        tracking_events = [
            {"status": "processing", "timestamp": order.get("timestamp", timestamp)},
            {"status": status, "timestamp": timestamp}
        ]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="TrackOrder",
            request={"order_number": order_number},
            response={
                "order_number": order_number,
                "status": status,
                "current_location": current_location,
                "estimated_delivery": order.get("estimated_delivery", self._calculate_delivery_date(timestamp)),
                "tracking_events": tracking_events
            }
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:5])

    def _extract_product_name(self, description: str) -> str:
        """Extract product name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:5])

    def _extract_quantity(self, description: str) -> int:
        """Extract quantity from description."""
        import re
        numbers = re.findall(r'\d+', description)
        return int(numbers[0]) if numbers else 1

    def _generate_search_results(self, query: str, context: Dict[str, Any]) -> List[Dict]:
        """Generate realistic search results."""
        num_products = random.randint(3, 8)
        products = []

        for i in range(num_products):
            products.append({
                "name": f"{query.title()} - Model {chr(65+i)}",
                "price": round(random.uniform(19.99, 299.99), 2),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "description": f"High-quality {query} product"
            })

        return products

    def _get_or_create_product(self, product_name: str, context: Dict[str, Any]) -> Dict:
        """Get or create product by name."""
        # Check if we have recently viewed this product
        for viewed in reversed(self.state["viewed_products"]):
            if viewed["product_name"].lower() == product_name.lower():
                # Return existing product with same attributes
                return {
                    "name": viewed["product_name"],
                    "price": round(random.uniform(19.99, 299.99), 2),
                    "rating": round(random.uniform(3.5, 5.0), 1),
                    "reviews": random.randint(10, 5000),
                    "description": f"High-quality {viewed['product_name']}",
                    "in_stock": True
                }

        # Create new product
        return {
            "name": product_name,
            "price": round(random.uniform(19.99, 299.99), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews": random.randint(10, 5000),
            "description": f"High-quality {product_name}",
            "in_stock": True
        }

    def _calculate_delivery_date(self, order_timestamp: str) -> str:
        """Calculate delivery date (2-5 days from order)."""
        dt = datetime.strptime(order_timestamp, "%Y-%m-%d %H:%M:%S")
        delivery_dt = dt + timedelta(days=random.randint(2, 5))
        return delivery_dt.strftime("%Y-%m-%d")

    def _generate_reviews(self, product_name: str, rating: float) -> List[Dict[str, Any]]:
        """Generate lightweight review samples."""
        review_templates = [
            "Works exactly as expected for daily use.",
            "Solid quality for the price.",
            "Setup was easy and it performs well.",
            "Decent product, but could be improved.",
            "Exceeded my expectations so far."
        ]
        reviews = []
        for _ in range(random.randint(3, 6)):
            review_rating = max(1, min(5, int(round(random.uniform(rating - 1, rating + 1)))))
            reviews.append({
                "reviewer": random.choice(["Alex", "Jamie", "Taylor", "Morgan", "Riley"]),
                "rating": review_rating,
                "text": random.choice(review_templates)
            })
        return reviews

    def _extract_order_number(self, description: str) -> str:
        """Extract order number from description."""
        import re
        match = re.search(r'AMZ\d{6,}', description)
        return match.group(0) if match else ""


class GoogleApp(BaseApp):
    """Google search app."""

    def __init__(self, user_id: str):
        super().__init__("Google", user_id)

    def _initialize_state(self) -> None:
        """Initialize Google state."""
        self.state = {
            "search_history": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Google API."""
        self.ensure_initialized()

        if api_name == "Search":
            return self._search(timestamp, description, context)
        elif api_name == "SearchNews":
            return self._search_news(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Google API: {api_name}")

    def _search(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle search."""
        query = self._extract_search_query(description)
        results = self._generate_search_results(query)

        self.state["search_history"].append(
            {"timestamp": timestamp, "query": query, "results_count": len(results)}
        )

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Search",
            request={"query": query},
            response={"results": results, "total_results": len(results)}
        )

    def _search_news(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle news search."""
        query = self._extract_search_query(description)
        base_date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        results = self._generate_news_results(query, base_date)

        self.state["search_history"].append(
            {"timestamp": timestamp, "query": query, "results_count": len(results), "type": "news"}
        )

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchNews",
            request={"query": query},
            response={"results": results, "total_results": len(results)}
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:6])

    def _generate_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate basic search results without URLs to avoid hallucination."""
        results = []
        sources = ["Wikipedia", "News article", "Academic paper", "Blog post", "Forum discussion", "Official site"]
        for idx in range(random.randint(3, 6)):
            results.append({
                "title": f"{query.title()} - Result {idx + 1}",
                "snippet": f"Relevant information about {query}. This result provides useful context and details.",
                "source": random.choice(sources)
            })
        return results

    def _generate_news_results(self, query: str, base_date: datetime) -> List[Dict[str, Any]]:
        """Generate basic news results."""
        results = []
        sources = ["Reuters", "AP News", "BBC", "The Verge", "Local News", "Financial Times"]
        for idx in range(random.randint(3, 6)):
            results.append({
                "title": f"{query.title()} - Update {idx + 1}",
                "snippet": f"Recent developments related to {query}.",
                "source": random.choice(sources),
                "date": (base_date - timedelta(days=idx)).strftime("%Y-%m-%d")
            })
        return results


class SpotifyApp(BaseApp):
    """Spotify music streaming app."""

    def __init__(self, user_id: str):
        super().__init__("Spotify", user_id)

    def _initialize_state(self) -> None:
        """Initialize Spotify state."""
        self.state = {
            "playlists": self._create_initial_playlists(),
            "recently_played": [],
            "favorite_genres": ["Pop", "Rock", "Electronic"]
        }

    def _create_initial_playlists(self) -> List[Dict]:
        """Create some initial playlists."""
        return [
            {
                "name": "My Favorites",
                "songs": [SpotifyDatabase.get_random_song() for _ in range(3)],
                "created_at": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "name": "Workout Mix",
                "songs": [SpotifyDatabase.get_random_song() for _ in range(3)],
                "created_at": datetime.now().strftime("%Y-%m-%d")
            }
        ]

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Spotify API."""
        self.ensure_initialized()

        if api_name == "SearchSongs":
            return self._search_songs(timestamp, description, context)
        elif api_name == "ShowArtist":
            return self._show_artist(timestamp, description, context)
        elif api_name == "PlaySong":
            return self._play_song(timestamp, description, context)
        elif api_name == "CreatePlaylist":
            return self._create_playlist(timestamp, description, context)
        elif api_name == "AddToPlaylist":
            return self._add_to_playlist(timestamp, description, context)
        elif api_name == "ShowPlaylists":
            return self._show_playlists(timestamp, description, context)
        elif api_name == "ShowRecentlyPlayed":
            return self._show_recently_played(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Spotify API: {api_name}")

    def _search_songs(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search for songs."""
        query = self._extract_search_query(description)
        songs = self._generate_song_results(query)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchSongs",
            request={"query": query},
            response={"songs": songs, "total_results": len(songs)}
        )

    def _show_artist(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show artist profile and top songs."""
        artist_name = self._extract_artist_name(description)
        songs = SpotifyDatabase.get_songs_by_artist(artist_name, limit=5)
        if not songs:
            songs = SpotifyDatabase.search_songs(artist_name, limit=5)
        genres = sorted({song.get("genre", "Unknown") for song in songs}) or ["Unknown"]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowArtist",
            request={"artist_name": artist_name},
            response={
                "artist": artist_name,
                "genres": genres,
                "top_songs": songs
            }
        )

    def _play_song(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Play a song."""
        song_title, artist = self._extract_song_info(description)
        song = self._get_or_create_song(song_title, artist, context)

        # Add to recently played
        self.state["recently_played"].insert(0, {
            "timestamp": timestamp,
            "song": song
        })

        # Keep only last 50
        self.state["recently_played"] = self.state["recently_played"][:50]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="PlaySong",
            request={"song_title": song_title, "artist": artist},
            response={
                "song": song,
                "status": "playing",
                "duration_seconds": song["duration"]
            }
        )

    def _create_playlist(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create a new playlist."""
        playlist_name = self._extract_playlist_name(description)
        playlist = {
            "name": playlist_name,
            "songs": [],
            "created_at": timestamp.split()[0]
        }
        self.state["playlists"].append(playlist)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreatePlaylist",
            request={"playlist_name": playlist_name},
            response={
                "playlist_name": playlist_name,
                "song_count": len(playlist["songs"]),
                "created_at": playlist["created_at"]
            }
        )

    def _add_to_playlist(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Add a song to an existing playlist."""
        playlist_name = self._extract_playlist_name(description)
        playlist = self._find_playlist(playlist_name)
        if not playlist:
            playlist = {
                "name": playlist_name,
                "songs": [],
                "created_at": timestamp.split()[0]
            }
            self.state["playlists"].append(playlist)

        song_title, artist = self._extract_song_info(description)
        song = self._get_or_create_song(song_title, artist, context)
        playlist["songs"].append(song)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="AddToPlaylist",
            request={"playlist_name": playlist_name, "song_title": song_title, "artist": artist},
            response={
                "playlist_name": playlist_name,
                "song_added": song,
                "song_count": len(playlist["songs"])
            }
        )

    def _show_playlists(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show user playlists."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowPlaylists",
            request={},
            response={
                "playlists": [
                    {
                        "name": playlist["name"],
                        "song_count": len(playlist.get("songs", [])),
                        "created_at": playlist.get("created_at", "")
                    }
                    for playlist in self.state["playlists"]
                ]
            }
        )

    def _show_recently_played(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View recently played songs."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowRecentlyPlayed",
            request={},
            response={
                "songs": [item["song"] for item in self.state["recently_played"][:20]]
            }
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "popular music"

    def _extract_song_info(self, description: str) -> tuple[str, str]:
        """Extract song title and artist from description."""
        import re
        # Try to extract from quotes
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            song_info = quoted[0][0] or quoted[0][1]
            # Try to split by 'by' or '-'
            if " by " in song_info.lower():
                parts = song_info.split(" by ", 1)
                return parts[0].strip(), parts[1].strip() if len(parts) > 1 else "Unknown Artist"
            elif " - " in song_info:
                parts = song_info.split(" - ", 1)
                return parts[0].strip(), parts[1].strip() if len(parts) > 1 else "Unknown Artist"
            return song_info, "Unknown Artist"

        # Fallback: use description
        words = description.split()
        return " ".join(words[:3]), "Unknown Artist"

    def _extract_artist_name(self, description: str) -> str:
        """Extract artist name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            artist = quoted[0][0] or quoted[0][1]
            if " by " in artist.lower():
                return artist.split(" by ", 1)[1].strip()
            return artist.strip()
        match = re.search(r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', description)
        if match:
            return match.group(1).strip()
        return "Unknown Artist"

    def _extract_playlist_name(self, description: str) -> str:
        """Extract playlist name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        match = re.search(r'playlist\s+([A-Za-z0-9 _-]+)', description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return random.choice(["New Playlist", "Favorites", "Daily Mix"])

    def _find_playlist(self, playlist_name: str) -> Optional[Dict[str, Any]]:
        """Find a playlist by name."""
        for playlist in self.state["playlists"]:
            if playlist["name"].lower() == playlist_name.lower():
                return playlist
        return None

    def _generate_song_results(self, query: str) -> List[Dict]:
        """Generate song search results using static database."""
        songs = SpotifyDatabase.search_songs(query, limit=8)
        return songs

    def _get_or_create_song(self, song_title: str, artist: str, context: Dict[str, Any]) -> Dict:
        """Get or create a song using static database."""
        # Check recently played for exact match
        for item in self.state["recently_played"]:
            song = item["song"]
            if (song.get("title", "").lower() == song_title.lower() and
                song.get("artist", "").lower() == artist.lower()):
                return song

        # Try to find in database
        desc_lower = song_title.lower() + " " + artist.lower()
        if "workout" in desc_lower or "energy" in desc_lower:
            songs = SpotifyDatabase.get_songs_by_genre("Hip Hop", limit=1)
            return songs[0] if songs else SpotifyDatabase.get_random_song()
        elif "chill" in desc_lower or "relax" in desc_lower:
            songs = SpotifyDatabase.get_songs_by_genre("Ambient", limit=1)
            return songs[0] if songs else SpotifyDatabase.get_random_song()
        elif "indie" in desc_lower:
            songs = SpotifyDatabase.get_songs_by_genre("Indie Rock", limit=1)
            return songs[0] if songs else SpotifyDatabase.get_random_song()
        else:
            return SpotifyDatabase.get_random_song()


class SimpleNoteApp(BaseApp):
    """Simple note-taking app."""

    def __init__(self, user_id: str):
        super().__init__("SimpleNote", user_id)

    def _initialize_state(self) -> None:
        """Initialize note state."""
        self.state = {
            "notes": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call SimpleNote API."""
        self.ensure_initialized()

        if api_name == "ShowNotes":
            return self._show_notes(timestamp, description, context)
        elif api_name == "ShowNote":
            return self._show_note(timestamp, description, context)
        elif api_name == "CreateNote":
            return self._create_note(timestamp, description, context)
        elif api_name == "EditNote":
            return self._edit_note(timestamp, description, context)
        elif api_name == "SearchNotes":
            return self._search_notes(timestamp, description, context)
        elif api_name == "TagNote":
            return self._tag_note(timestamp, description, context)
        else:
            raise ValueError(f"Unknown SimpleNote API: {api_name}")

    def _show_notes(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show all notes."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowNotes",
            request={},
            response={
                "notes": [
                    {
                        "title": note["title"],
                        "preview": note["content"][:50] + ("..." if len(note["content"]) > 50 else ""),
                        "tags": note.get("tags", []),
                        "created_at": note["created_at"]
                    }
                    for note in self.state["notes"]
                ],
                "total_notes": len(self.state["notes"])
            }
        )

    def _show_note(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show a specific note by title."""
        note_title = self._extract_note_title(description)

        # Find note by title
        note = None
        for n in self.state["notes"]:
            if n["title"].lower() == note_title.lower():
                note = n
                break

        # If not found, use most recent or create placeholder
        if not note:
            if self.state["notes"]:
                note = self.state["notes"][-1]
            else:
                note = {
                    "title": note_title,
                    "content": "This is a placeholder note.",
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "tags": []
                }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowNote",
            request={"note_title": note_title},
            response={
                "title": note["title"],
                "content": note["content"],
                "tags": note.get("tags", []),
                "created_at": note.get("created_at", timestamp),
                "updated_at": note.get("updated_at", note.get("created_at", timestamp))
            }
        )

    def _create_note(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create a new note."""
        title, content = self._extract_note_content(description)

        note = {
            "title": title,
            "content": content,
            "created_at": timestamp,
            "updated_at": timestamp,
            "tags": self._extract_tags(description)
        }

        self.state["notes"].append(note)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateNote",
            request={"title": title, "content": content},
            response={
                "title": note["title"],
                "content": note["content"],
                "tags": note["tags"],
                "created_at": note["created_at"]
            }
        )

    def _edit_note(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Edit an existing note."""
        note_title = self._extract_note_title(description)
        content_update = description
        note = None
        for existing in self.state["notes"]:
            if existing["title"].lower() == note_title.lower():
                note = existing
                break

        if not note:
            note = {
                "title": note_title,
                "content": "",
                "created_at": timestamp,
                "tags": []
            }
            self.state["notes"].append(note)

        note["content"] = content_update
        note["updated_at"] = timestamp

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="EditNote",
            request={"note_title": note_title, "content": content_update},
            response={
                "title": note["title"],
                "content": note["content"],
                "tags": note.get("tags", []),
                "updated_at": note["updated_at"]
            }
        )

    def _search_notes(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search notes by query."""
        query = self._extract_search_query(description)
        query_lower = query.lower()
        results = [
            note for note in self.state["notes"]
            if query_lower in note["title"].lower()
            or query_lower in note["content"].lower()
            or query_lower in " ".join(note.get("tags", [])).lower()
        ]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchNotes",
            request={"query": query},
            response={
                "notes": results[:10],
                "total_results": len(results)
            }
        )

    def _tag_note(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Add tags to a note."""
        note_title = self._extract_note_title(description)
        tags = self._extract_tags(description)
        note = None
        for existing in self.state["notes"]:
            if existing["title"].lower() == note_title.lower():
                note = existing
                break

        if not note:
            note = {
                "title": note_title,
                "content": "",
                "created_at": timestamp,
                "updated_at": timestamp,
                "tags": []
            }
            self.state["notes"].append(note)

        note["tags"] = sorted(set(note.get("tags", []) + tags))
        note["updated_at"] = timestamp

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="TagNote",
            request={"note_title": note_title, "tags": tags},
            response={
                "title": note["title"],
                "tags": note["tags"],
                "updated_at": note["updated_at"]
            }
        )

    def _extract_note_title(self, description: str) -> str:
        """Extract note title from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:5])

    def _extract_note_content(self, description: str) -> tuple[str, str]:
        """Extract note title and content from description."""
        words = description.split()
        title = " ".join(words[:5])
        content = description
        return title, content

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "note"

    def _extract_tags(self, description: str) -> List[str]:
        """Extract tags from description."""
        tags = [word.lstrip("#") for word in description.split() if word.startswith("#")]
        if tags:
            return tags
        return []


class MessageApp(BaseApp):
    """Messaging app."""

    def __init__(self, user_id: str):
        super().__init__("Message", user_id)

    def _initialize_state(self) -> None:
        """Initialize messaging state."""
        self.state = {
            "conversations": {},  # contact_name -> list of messages
            "contacts": ["Alice", "Bob", "Carol", "Dave", "Emma"],
            "groups": {}  # group_name -> group info
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Message API."""
        self.ensure_initialized()

        if api_name == "SendMessage":
            return self._send_message(timestamp, description, context)
        elif api_name == "GetMessages":
            return self._get_messages(timestamp, description, context)
        elif api_name == "SearchMessages":
            return self._search_messages(timestamp, description, context)
        elif api_name == "CreateGroup":
            return self._create_group(timestamp, description, context)
        elif api_name == "ReactToMessage":
            return self._react_to_message(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Message API: {api_name}")

    def _send_message(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Send a message."""
        recipient, message_text = self._extract_message_details(description)

        message = {
            "message_id": str(uuid.uuid4()),
            "from": self.user_id,
            "to": recipient,
            "text": message_text,
            "timestamp": timestamp,
            "status": "sent",
            "reactions": []
        }

        # Add to conversation
        if recipient not in self.state["conversations"]:
            self.state["conversations"][recipient] = []
        self.state["conversations"][recipient].append(message)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SendMessage",
            request={"to": recipient, "text": message_text},
            response=message
        )

    def _get_messages(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Get messages from a conversation thread."""
        contact_name = self._extract_contact_name(description)
        messages = self.state["conversations"].get(contact_name, [])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetMessages",
            request={"contact_name": contact_name},
            response={
                "contact_name": contact_name,
                "messages": messages[-20:],  # Last 20 messages
                "total_count": len(messages)
            }
        )

    def _search_messages(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search messages."""
        query = self._extract_search_query(description)

        # Simple search through all conversations
        results = []
        for contact, messages in self.state["conversations"].items():
            for msg in messages:
                if query.lower() in msg["text"].lower():
                    results.append(msg)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchMessages",
            request={"query": query},
            response={
                "messages": results[:10],  # Return up to 10 results
                "total_count": len(results)
            }
        )

    def _create_group(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create a group chat."""
        group_name = self._extract_group_name(description)
        members = random.sample(self.state["contacts"], min(3, len(self.state["contacts"])))
        group_id = str(uuid.uuid4())

        self.state["groups"][group_name] = {
            "group_id": group_id,
            "group_name": group_name,
            "members": members
        }
        self.state["conversations"].setdefault(group_name, [])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateGroup",
            request={"group_name": group_name, "members": members},
            response={
                "group_id": group_id,
                "group_name": group_name,
                "members": members
            }
        )

    def _react_to_message(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """React to a message."""
        contact_name = self._extract_contact_name(description)
        reaction = self._extract_reaction(description)
        messages = self.state["conversations"].get(contact_name, [])
        target_message = messages[-1] if messages else None
        message_id = target_message["message_id"] if target_message else str(uuid.uuid4())

        if target_message:
            target_message.setdefault("reactions", []).append(reaction)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ReactToMessage",
            request={"message_id": message_id, "reaction": reaction},
            response={"message_id": message_id, "reaction": reaction, "status": "added"}
        )

    def _extract_contact_name(self, description: str) -> str:
        """Extract contact name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        # Fallback to random contact
        return random.choice(self.state["contacts"]) if self.state["contacts"] else "Friend"

    def _extract_message_details(self, description: str) -> tuple[str, str]:
        """Extract recipient and message from description."""
        import re
        # Try to find quoted message
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        message_text = quoted[0][0] or quoted[0][1] if quoted else description[:100]

        # Extract recipient - look for "to X" pattern
        recipient_match = re.search(r'\bto\s+([A-Z][a-z]+)', description)
        if recipient_match:
            recipient = recipient_match.group(1)
        else:
            recipient = random.choice(self.state["contacts"]) if self.state["contacts"] else "Friend"

        return recipient, message_text

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "search query"

    def _extract_group_name(self, description: str) -> str:
        """Extract group name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        match = re.search(r'group\s+([A-Za-z0-9 _-]+)', description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return random.choice(["Project Team", "Friends", "Family"])

    def _extract_reaction(self, description: str) -> str:
        """Extract reaction emoji or text."""
        for token in description.split():
            if token in ["👍", "👎", "❤️", "😂", "🎉"]:
                return token
        return random.choice(["👍", "❤️", "😂"])


class FitnessApp(BaseApp):
    """Fitness tracking app."""

    def __init__(self, user_id: str):
        super().__init__("Fitness", user_id)

    def _initialize_state(self) -> None:
        """Initialize fitness state."""
        self.state = {
            "workouts": [],
            "daily_stats": {},  # date -> stats
            "goals": {
                "daily_steps": 10000,
                "weekly_workouts": 5,
                "weekly_active_minutes": 150
            },
            "user_baseline": None  # Will be initialized on first use
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Fitness API."""
        self.ensure_initialized()

        if api_name == "LogWorkout":
            return self._log_workout(timestamp, description, context)
        elif api_name == "LogActivity":
            return self._log_activity(timestamp, description, context)
        elif api_name == "ShowDailyStats":
            return self._show_daily_stats(timestamp, description, context)
        elif api_name == "ShowWeeklyStats":
            return self._show_weekly_stats(timestamp, description, context)
        elif api_name == "SetGoal":
            return self._set_goal(timestamp, description, context)
        elif api_name == "ShowProgress":
            return self._show_progress(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Fitness API: {api_name}")

    def _log_workout(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Log a workout."""
        # Initialize baseline if needed
        if self.state["user_baseline"] is None:
            self._initialize_user_baseline(context)

        activity_type, duration, intensity = self._extract_workout_details(description)

        workout = {
            "activity_type": activity_type,
            "duration_minutes": duration,
            "intensity": intensity,
            "calories_burned": self._calculate_calories(activity_type, duration, intensity),
            "heart_rate_avg": self._generate_heart_rate(intensity),
            "timestamp": timestamp
        }

        self.state["workouts"].append(workout)

        # Update daily stats
        date = timestamp.split()[0]
        if date not in self.state["daily_stats"]:
            self.state["daily_stats"][date] = {
                "date": date,
                "steps": 0,
                "active_minutes": 0,
                "calories": 0,
                "workouts": []
            }

        self.state["daily_stats"][date]["active_minutes"] += duration
        self.state["daily_stats"][date]["calories"] += workout["calories_burned"]
        self.state["daily_stats"][date]["workouts"].append(workout)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="LogWorkout",
            request={
                "activity_type": activity_type,
                "duration_minutes": duration,
                "intensity": intensity
            },
            response=workout
        )

    def _log_activity(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Log daily activity metrics."""
        date = timestamp.split()[0]
        steps = self._extract_number(description, default=random.randint(3000, 12000))
        calories = self._extract_number(description, default=random.randint(1800, 2500))
        active_minutes = random.randint(20, 120)

        if date not in self.state["daily_stats"]:
            self.state["daily_stats"][date] = {
                "date": date,
                "steps": 0,
                "active_minutes": 0,
                "calories": 0,
                "workouts": []
            }

        self.state["daily_stats"][date]["steps"] = steps
        self.state["daily_stats"][date]["calories"] = calories
        self.state["daily_stats"][date]["active_minutes"] = active_minutes

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="LogActivity",
            request={"steps": steps, "calories": calories, "active_minutes": active_minutes},
            response={
                "date": date,
                "steps": steps,
                "calories": calories,
                "active_minutes": active_minutes
            }
        )

    def _show_daily_stats(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show today's fitness stats."""
        date = timestamp.split()[0]

        if date not in self.state["daily_stats"]:
            # Generate daily stats
            self.state["daily_stats"][date] = {
                "date": date,
                "steps": random.randint(3000, 12000),
                "active_minutes": random.randint(20, 90),
                "calories": random.randint(1800, 2500),
                "workouts": []
            }

        stats = self.state["daily_stats"][date]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowDailyStats",
            request={},
            response=stats
        )

    def _show_weekly_stats(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show weekly fitness stats."""
        # Aggregate last 7 days
        total_workouts = len(self.state["workouts"][-7:])
        total_active_minutes = sum(w["duration_minutes"] for w in self.state["workouts"][-7:])

        weekly_stats = {
            "period": "last_7_days",
            "total_workouts": total_workouts,
            "total_active_minutes": total_active_minutes,
            "avg_daily_steps": random.randint(6000, 10000)
        }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowWeeklyStats",
            request={},
            response=weekly_stats
        )

    def _set_goal(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Set or update a fitness goal."""
        goal_type = self._extract_goal_type(description)
        target = self._extract_number(description, default=self.state["goals"].get(goal_type, 100))
        self.state["goals"][goal_type] = target

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SetGoal",
            request={"goal_type": goal_type, "target": target},
            response={"goals": self.state["goals"].copy()}
        )

    def _show_progress(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show progress toward goals."""
        date = timestamp.split()[0]
        stats = self.state["daily_stats"].get(date, {
            "steps": 0,
            "active_minutes": 0,
            "calories": 0
        })

        progress = {
            "daily_steps": {
                "current": stats.get("steps", 0),
                "target": self.state["goals"].get("daily_steps", 10000)
            },
            "weekly_workouts": {
                "current": len(self.state["workouts"][-7:]),
                "target": self.state["goals"].get("weekly_workouts", 5)
            },
            "weekly_active_minutes": {
                "current": sum(w["duration_minutes"] for w in self.state["workouts"][-7:]),
                "target": self.state["goals"].get("weekly_active_minutes", 150)
            }
        }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowProgress",
            request={},
            response={"progress": progress}
        )

    def _initialize_user_baseline(self, context: Dict[str, Any]) -> None:
        """Initialize user fitness baseline based on context using LLM data generator."""
        # Use LLM data generator for personalized baseline
        data_gen = get_data_generator()
        user_profile = context.get("user_profile", {})

        # Extract description from context if available
        description = context.get("description", "")

        self.state["user_baseline"] = data_gen.generate_fitness_baseline(
            user_profile=user_profile,
            description=description
        )

    def _extract_workout_details(self, description: str) -> tuple[str, int, str]:
        """Extract workout type, duration, and intensity."""
        # Parse description for workout details
        desc_lower = description.lower()

        # Determine workout type
        if "run" in desc_lower or "jog" in desc_lower:
            workout_type = "running"
        elif "bike" in desc_lower or "cycle" in desc_lower:
            workout_type = "cycling"
        elif "swim" in desc_lower:
            workout_type = "swimming"
        elif "strength" in desc_lower or "weight" in desc_lower:
            workout_type = "strength_training"
        elif "yoga" in desc_lower:
            workout_type = "yoga"
        else:
            workout_type = random.choice(["running", "cycling", "strength_training", "other"])

        # Duration
        duration = random.randint(20, 90)

        # Intensity
        if "intense" in desc_lower or "hard" in desc_lower:
            intensity = "high"
        elif "easy" in desc_lower or "light" in desc_lower:
            intensity = "low"
        else:
            intensity = "moderate"

        return workout_type, duration, intensity

    def _extract_goal_type(self, description: str) -> str:
        """Extract goal type from description."""
        desc_lower = description.lower()
        if "steps" in desc_lower:
            return "daily_steps"
        if "workout" in desc_lower:
            return "weekly_workouts"
        if "active" in desc_lower or "minutes" in desc_lower:
            return "weekly_active_minutes"
        return "daily_steps"

    def _extract_number(self, description: str, default: int = 0) -> int:
        """Extract a number from description."""
        import re
        match = re.search(r'\b(\d{2,6})\b', description)
        return int(match.group(1)) if match else default

    def _calculate_calories(self, workout_type: str, duration: int, intensity: str) -> int:
        """Calculate calories burned."""
        base_rate = {
            "running": 10,
            "cycling": 8,
            "swimming": 9,
            "strength_training": 6,
            "yoga": 3,
            "other": 5
        }.get(workout_type, 5)

        intensity_multiplier = {"low": 0.7, "moderate": 1.0, "high": 1.3}.get(intensity, 1.0)

        return int(base_rate * duration * intensity_multiplier)

    def _generate_heart_rate(self, intensity: str) -> int:
        """Generate average heart rate for workout."""
        if self.state["user_baseline"]:
            resting_hr = self.state["user_baseline"]["avg_resting_heart_rate"]
        else:
            resting_hr = 65

        intensity_range = {
            "low": (resting_hr + 20, resting_hr + 40),
            "moderate": (resting_hr + 40, resting_hr + 60),
            "high": (resting_hr + 60, resting_hr + 80)
        }.get(intensity, (resting_hr + 30, resting_hr + 50))

        return random.randint(*intensity_range)


class CalendarApp(BaseApp):
    """Calendar scheduling app."""

    def __init__(self, user_id: str):
        super().__init__("Calendar", user_id)

    def _initialize_state(self) -> None:
        """Initialize calendar state."""
        self.state = {
            "events": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Calendar API."""
        self.ensure_initialized()

        if api_name == "CreateEvent":
            return self._create_event(timestamp, description, context)
        elif api_name == "ShowEvents":
            return self._show_events(timestamp, description, context)
        elif api_name == "EditEvent":
            return self._edit_event(timestamp, description, context)
        elif api_name == "SetReminder":
            return self._set_reminder(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Calendar API: {api_name}")

    def _create_event(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create a calendar event."""
        title = self._extract_event_title(description)
        date = timestamp.split()[0]
        start_time, end_time = self._extract_time_range(timestamp, description)
        location = self._extract_location(description)

        event = {
            "event_id": str(uuid.uuid4()),
            "title": title,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
            "reminder": ""
        }
        self.state["events"].append(event)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateEvent",
            request={"title": title, "date": date, "start_time": start_time, "end_time": end_time},
            response=event
        )

    def _show_events(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show calendar events."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowEvents",
            request={},
            response={
                "events": self.state["events"],
                "total_events": len(self.state["events"])
            }
        )

    def _edit_event(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Edit a calendar event."""
        title = self._extract_event_title(description)
        event = self._find_event_by_title(title)
        if not event and self.state["events"]:
            event = self.state["events"][-1]

        if not event:
            event = {
                "event_id": str(uuid.uuid4()),
                "title": title,
                "date": timestamp.split()[0],
                "start_time": timestamp.split()[1][:5],
                "end_time": self._add_minutes(timestamp, 60),
                "location": "",
                "reminder": ""
            }
            self.state["events"].append(event)

        start_time, end_time = self._extract_time_range(timestamp, description)
        if start_time:
            event["start_time"] = start_time
            event["end_time"] = end_time
        if title:
            event["title"] = title

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="EditEvent",
            request={"event_id": event["event_id"], "updates": event},
            response={"event": event}
        )

    def _set_reminder(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Set reminder for a calendar event."""
        title = self._extract_event_title(description)
        event = self._find_event_by_title(title)
        if not event and self.state["events"]:
            event = self.state["events"][-1]

        reminder_minutes = self._extract_number(description, default=30)
        if event:
            event["reminder"] = f"{reminder_minutes} minutes before"

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SetReminder",
            request={"event_id": event["event_id"] if event else "", "reminder_minutes": reminder_minutes},
            response={"event_id": event["event_id"] if event else "", "reminder_minutes": reminder_minutes}
        )

    def _extract_event_title(self, description: str) -> str:
        """Extract event title from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:6]) if words else "New Event"

    def _extract_time_range(self, timestamp: str, description: str) -> tuple[str, str]:
        """Extract time range from description or fallback to timestamp."""
        import re
        times = re.findall(r'\b(\d{1,2}:\d{2})\b', description)
        if len(times) >= 2:
            return times[0].zfill(5), times[1].zfill(5)
        if len(times) == 1:
            start_time = times[0].zfill(5)
            end_time = self._add_minutes(f"{timestamp.split()[0]} {start_time}:00", 60)
            return start_time, end_time
        start_time = timestamp.split()[1][:5]
        end_time = self._add_minutes(timestamp, 60)
        return start_time, end_time

    def _add_minutes(self, timestamp: str, minutes: int) -> str:
        """Add minutes to a timestamp or time string."""
        if len(timestamp.split()) == 2:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            dt = datetime.strptime(f"2000-01-01 {timestamp}", "%Y-%m-%d %H:%M:%S")
        return (dt + timedelta(minutes=minutes)).strftime("%H:%M")

    def _extract_location(self, description: str) -> str:
        """Extract location from description."""
        import re
        match = re.search(r'\bat\s+([A-Za-z0-9 ,.-]+)', description)
        if match:
            return match.group(1).strip()
        return "unspecified"

    def _find_event_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Find an event by title."""
        for event in self.state["events"]:
            if event["title"].lower() == title.lower():
                return event
        return None

    def _extract_number(self, description: str, default: int = 0) -> int:
        """Extract a number from description."""
        import re
        match = re.search(r'\b(\d{1,4})\b', description)
        return int(match.group(1)) if match else default


class FinanceApp(BaseApp):
    """Finance management app."""

    def __init__(self, user_id: str):
        super().__init__("Finance", user_id)

    def _initialize_state(self) -> None:
        """Initialize finance state."""
        self.state = {
            "accounts": [
                {"account_id": "acc_checking", "name": "Checking", "balance": 2500.0, "type": "checking"},
                {"account_id": "acc_savings", "name": "Savings", "balance": 10000.0, "type": "savings"},
                {"account_id": "acc_credit", "name": "Credit Card", "balance": -350.0, "type": "credit"}
            ],
            "transactions": [],
            "budgets": {},
            "financial_goals": [],
            "investments": [
                {"symbol": "VTI", "shares": 10, "price": 220.0},
                {"symbol": "AAPL", "shares": 5, "price": 180.0}
            ]
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Finance API."""
        self.ensure_initialized()

        if api_name == "ShowAccounts":
            return self._show_accounts(timestamp, description, context)
        elif api_name == "ShowTransactions":
            return self._show_transactions(timestamp, description, context)
        elif api_name == "CreateBudget":
            return self._create_budget(timestamp, description, context)
        elif api_name == "LogExpense":
            return self._log_expense(timestamp, description, context)
        elif api_name == "ShowBudgetProgress":
            return self._show_budget_progress(timestamp, description, context)
        elif api_name == "SetFinancialGoal":
            return self._set_financial_goal(timestamp, description, context)
        elif api_name == "ShowInvestments":
            return self._show_investments(timestamp, description, context)
        elif api_name == "TransferMoney":
            return self._transfer_money(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Finance API: {api_name}")

    def _show_accounts(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show account balances."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowAccounts",
            request={},
            response={"accounts": self.state["accounts"]}
        )

    def _show_transactions(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show recent transactions."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowTransactions",
            request={},
            response={
                "transactions": self.state["transactions"][-20:],
                "total_transactions": len(self.state["transactions"])
            }
        )

    def _create_budget(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create or update a budget."""
        categories = self._extract_budget_categories(description)
        for category, limit in categories.items():
            self.state["budgets"][category] = limit

        budgets = [{"category": k, "limit": v} for k, v in self.state["budgets"].items()]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateBudget",
            request={"categories": budgets},
            response={"budgets": budgets}
        )

    def _log_expense(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Log an expense transaction."""
        amount = self._extract_amount(description, default=round(random.uniform(5.0, 120.0), 2))
        category = self._extract_category(description)
        merchant = self._extract_merchant(description)
        account = self._select_account("checking")

        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "amount": -abs(amount),
            "category": category,
            "merchant": merchant,
            "timestamp": timestamp,
            "account_id": account["account_id"]
        }
        self.state["transactions"].append(transaction)
        account["balance"] -= abs(amount)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="LogExpense",
            request={"amount": amount, "category": category, "merchant": merchant},
            response={"transaction": transaction}
        )

    def _show_budget_progress(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show budget progress by category."""
        spending = {}
        for txn in self.state["transactions"]:
            category = txn["category"]
            spending[category] = spending.get(category, 0) + abs(txn["amount"])

        categories = []
        for category, limit in self.state["budgets"].items():
            categories.append({
                "category": category,
                "spent": round(spending.get(category, 0), 2),
                "limit": limit
            })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowBudgetProgress",
            request={},
            response={"categories": categories}
        )

    def _set_financial_goal(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Set a financial goal."""
        goal_name = self._extract_goal_name(description)
        target_amount = self._extract_amount(description, default=5000.0)

        goal = {
            "goal_id": str(uuid.uuid4()),
            "goal_name": goal_name,
            "target_amount": target_amount,
            "created_at": timestamp
        }
        self.state["financial_goals"].append(goal)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SetFinancialGoal",
            request={"goal_name": goal_name, "target_amount": target_amount},
            response={"goal": goal}
        )

    def _show_investments(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show investment portfolio."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowInvestments",
            request={},
            response={"investments": self.state["investments"]}
        )

    def _transfer_money(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Transfer money between accounts."""
        amount = self._extract_amount(description, default=100.0)
        from_account = self._select_account("checking")
        to_account = self._select_account("savings")

        from_account["balance"] -= amount
        to_account["balance"] += amount

        transfer = {
            "transfer_id": str(uuid.uuid4()),
            "from_account": from_account["name"],
            "to_account": to_account["name"],
            "amount": amount,
            "timestamp": timestamp
        }

        self.state["transactions"].append({
            "transaction_id": transfer["transfer_id"],
            "amount": -amount,
            "category": "transfer",
            "merchant": f"Transfer to {to_account['name']}",
            "timestamp": timestamp,
            "account_id": from_account["account_id"]
        })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="TransferMoney",
            request={"from_account": from_account["name"], "to_account": to_account["name"], "amount": amount},
            response={"transfer": transfer}
        )

    def _extract_amount(self, description: str, default: float = 0.0) -> float:
        """Extract a monetary amount."""
        import re
        match = re.search(r'\$?(\d+(?:\.\d+)?)', description)
        return float(match.group(1)) if match else default

    def _extract_category(self, description: str) -> str:
        """Extract expense category."""
        categories = ["groceries", "transportation", "dining", "utilities", "entertainment", "health"]
        for category in categories:
            if category in description.lower():
                return category
        return random.choice(categories)

    def _extract_merchant(self, description: str) -> str:
        """Extract merchant name."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return random.choice(["Grocery Store", "Cafe", "Gas Station", "Online Retailer"])

    def _extract_budget_categories(self, description: str) -> Dict[str, float]:
        """Extract budget categories and limits."""
        categories = {}
        if "food" in description.lower():
            categories["dining"] = 300.0
        if "rent" in description.lower():
            categories["housing"] = 1200.0
        if not categories:
            categories = {"groceries": 400.0, "transportation": 200.0}
        return categories

    def _extract_goal_name(self, description: str) -> str:
        """Extract goal name."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return random.choice(["Emergency Fund", "Vacation", "New Laptop"])

    def _select_account(self, account_type: str) -> Dict[str, Any]:
        """Select account by type."""
        for account in self.state["accounts"]:
            if account["type"] == account_type:
                return account
        return self.state["accounts"][0]


class LLMApp(BaseApp):
    """LLM chat app."""

    def __init__(self, user_id: str):
        super().__init__("LLM", user_id)

    def _initialize_state(self) -> None:
        """Initialize LLM state."""
        self.state = {
            "conversation_history": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call LLM API."""
        self.ensure_initialized()

        if api_name == "Chat":
            return self._chat(timestamp, description, context)
        else:
            raise ValueError(f"Unknown LLM API: {api_name}")

    def _chat(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle chat - returns full conversation turn."""
        # Extract user message from description
        user_message = self._extract_user_message(description)

        # Generate AI response based on description
        ai_response = self._generate_ai_response(user_message, description, context)

        # Create conversation turn
        conversation_turn = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response}
        ]

        # Add to conversation history
        self.state["conversation_history"].append({
            "timestamp": timestamp,
            "user_message": user_message,
            "ai_response": ai_response
        })

        # Generate description of the conversation
        conv_description = self._generate_conversation_description(user_message, ai_response)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Chat",
            request={"message": user_message},
            response={
                "conversation": conversation_turn,
                "description": conv_description
            }
        )

    def _extract_user_message(self, description: str) -> str:
        """Extract user message from description."""
        import re
        # Try to extract from quotes
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        # Use description as is
        return description

    def _generate_ai_response(self, user_message: str, description: str, context: Dict[str, Any]) -> str:
        """Generate a plausible AI response using LLM data generator."""
        # Use LLM data generator for chat responses
        data_gen = get_data_generator()
        user_profile = context.get("user_profile", {})

        response = data_gen.generate_llm_chat_response(
            user_message=user_message,
            conversation_history=self.state["conversation_history"],
            user_profile=user_profile
        )

        return response

    def _generate_conversation_description(self, user_message: str, ai_response: str) -> str:
        """Generate a brief description of the conversation."""
        # Extract key topic from user message
        user_msg_short = user_message[:50] + "..." if len(user_message) > 50 else user_message
        ai_resp_short = ai_response[:50] + "..." if len(ai_response) > 50 else ai_response

        return f"User asked about: {user_msg_short}. Assistant provided: {ai_resp_short}"


class AppRegistry:
    """
    Registry for all app instances.

    Maintains one instance per (app_name, user_id) to ensure state consistency
    across all domains and time windows.
    """

    def __init__(self):
        self.apps: Dict[tuple[str, str], BaseApp] = {}

    def get_app(self, app_name: str, user_id: str) -> BaseApp:
        """Get or create an app instance for this user."""
        key = (app_name, user_id)

        if key not in self.apps:
            # Create new app instance
            if app_name == "Amazon":
                self.apps[key] = AmazonApp(user_id)
            elif app_name == "Google":
                self.apps[key] = GoogleApp(user_id)
            elif app_name == "Spotify":
                self.apps[key] = SpotifyApp(user_id)
            elif app_name == "SimpleNote":
                self.apps[key] = SimpleNoteApp(user_id)
            elif app_name == "LLM":
                self.apps[key] = LLMApp(user_id)
            elif app_name == "Message":
                self.apps[key] = MessageApp(user_id)
            elif app_name == "Fitness":
                self.apps[key] = FitnessApp(user_id)
            elif app_name == "Calendar":
                self.apps[key] = CalendarApp(user_id)
            elif app_name == "Finance":
                self.apps[key] = FinanceApp(user_id)
            else:
                raise ValueError(f"Unknown app: {app_name}")

        return self.apps[key]

    def call_api(
        self,
        app_name: str,
        api_name: str,
        user_id: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """
        Call an app API.

        This is the main entry point for generating app logs.
        """
        app = self.get_app(app_name, user_id)
        return app.call_api(api_name, timestamp, description, context)
