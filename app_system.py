"""
App System for generating realistic app logs with stateful consistency.

Each app maintains state about the user to ensure consistency across:
- Different domains calling the same app
- Multiple calls within the same domain

Apps implemented based on app_catalog.py and app_models.py definitions.
"""

from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from mem_bench.behavior_and_conversation.app_data_sources import (
    SpotifyDatabase,
    get_data_generator
)
from mem_bench.behavior_and_conversation.app_models import (
    # Amazon
    AmazonState, AmazonProduct, AmazonOrder,
    # Spotify
    SpotifyState, SpotifyPlaylist, SpotifySong, SpotifyPlayHistory,
    # Fitbit
    FitbitState, FitbitGoal, FitbitWorkout, FitbitDailySync,
    # Chase
    ChaseState, ChaseAccount, ChaseTransaction,
    # Robinhood
    RobinhoodState, RobinhoodHolding, RobinhoodTransaction,
    # WhatsApp
    WhatsAppState, WhatsAppMessage,
    # Gmail
    GmailState, GmailEmail,
    # LinkedIn
    LinkedInState, LinkedInExperience, LinkedInPost,
    # Notion
    NotionState, NotionPage, NotionDatabaseEntry,
    # Netflix
    NetflixState, NetflixTitle, NetflixViewHistory,
    # Goodreads
    GoodreadsState, GoodreadsBook, GoodreadsShelfEntry, GoodreadsReview,
    # Instagram
    InstagramState, InstagramPost,
    # LLM
    LLMState, LLMConversation, LLMMessage,
    # Google
    GoogleState, GoogleSearchHistory, GoogleSearchResult,
    # Enums
    AssetType, TransactionType, ChaseTransactionType, ChaseAccountType, ChaseCategory,
    MessageType, MediaType, ContentType, InstagramContentType, NetflixPlan, BookShelf,
    ChatRole, PlayingStatus, WorkoutIntensity, FitbitGoalType, ThumbsRating,
)


# Maximum number of history entries to keep per list to avoid state explosion
MAX_HISTORY_LENGTH = 20
@dataclass
class AppLogEntry:
    """A single app log entry."""
    timestamp: str  # "YYYY-MM-DD HH:MM:SS"
    app_name: str
    api_name: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _model_to_schema(model_class) -> Dict[str, Any]:
    """Convert a Pydantic model to JSON schema dict."""
    if hasattr(model_class, "model_json_schema"):
        return model_class.model_json_schema()
    elif hasattr(model_class, "schema"):
        return model_class.schema()
    return {}


def _build_api_schemas() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build API schemas from app_models.py Pydantic models."""
    from mem_bench.behavior_and_conversation import app_models
    
    schemas: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    # Amazon APIs
    schemas["Amazon"] = {
        "SearchProducts": {
            "input": _model_to_schema(app_models.SearchProductsInput),
            "output": _model_to_schema(app_models.SearchProductsOutput),
        },
        "ShowProduct": {
            "input": _model_to_schema(app_models.ShowProductInput),
            "output": _model_to_schema(app_models.ShowProductOutput),
        },
        "AddToCart": {
            "input": _model_to_schema(app_models.AddToCartInput),
            "output": _model_to_schema(app_models.AddToCartOutput),
        },
        "ShowCart": {
            "input": _model_to_schema(app_models.ShowCartInput),
            "output": _model_to_schema(app_models.ShowCartOutput),
        },
        "ShowWishlist": {
            "input": _model_to_schema(app_models.ShowWishlistInput),
            "output": _model_to_schema(app_models.ShowWishlistOutput),
        },
        "Checkout": {
            "input": _model_to_schema(app_models.CheckoutInput),
            "output": _model_to_schema(app_models.CheckoutOutput),
        },
    }
    
    # Spotify APIs
    schemas["Spotify"] = {
        "SearchSongs": {
            "input": _model_to_schema(app_models.SearchSongsInput),
            "output": _model_to_schema(app_models.SearchSongsOutput),
        },
        "PlaySong": {
            "input": _model_to_schema(app_models.PlaySongInput),
            "output": _model_to_schema(app_models.PlaySongOutput),
        },
        "AddToPlaylist": {
            "input": _model_to_schema(app_models.AddToPlaylistInput),
            "output": _model_to_schema(app_models.AddToPlaylistOutput),
        },
        "FollowArtist": {
            "input": _model_to_schema(app_models.FollowArtistInput),
            "output": _model_to_schema(app_models.FollowArtistOutput),
        },
    }
    
    # Fitbit APIs
    schemas["Fitbit"] = {
        "LogWorkout": {
            "input": _model_to_schema(app_models.LogWorkoutInput),
            "output": _model_to_schema(app_models.LogWorkoutOutput),
        },
        "SyncDevice": {
            "input": _model_to_schema(app_models.SyncDeviceInput),
            "output": _model_to_schema(app_models.SyncDeviceOutput),
        },
        "SetGoals": {
            "input": _model_to_schema(app_models.SetGoalsInput),
            "output": _model_to_schema(app_models.SetGoalsOutput),
        },
    }
    
    # Chase APIs
    schemas["Chase"] = {
        "GetBalance": {
            "input": _model_to_schema(app_models.GetBalanceInput),
            "output": _model_to_schema(app_models.GetBalanceOutput),
        },
        "GetTransactions": {
            "input": _model_to_schema(app_models.GetTransactionsInput),
            "output": _model_to_schema(app_models.GetTransactionsOutput),
        },
        "SearchTransactions": {
            "input": _model_to_schema(app_models.SearchTransactionsInput),
            "output": _model_to_schema(app_models.SearchTransactionsOutput),
        },
        "TransferMoney": {
            "input": _model_to_schema(app_models.TransferMoneyInput),
            "output": _model_to_schema(app_models.TransferMoneyOutput),
        },
        "PayBill": {
            "input": _model_to_schema(app_models.PayBillInput),
            "output": _model_to_schema(app_models.PayBillOutput),
        },
    }
    
    # Robinhood APIs
    schemas["Robinhood"] = {
        "GetPortfolio": {
            "input": _model_to_schema(app_models.GetPortfolioInput),
            "output": _model_to_schema(app_models.GetPortfolioOutput),
        },
        "GetWatchlist": {
            "input": _model_to_schema(app_models.GetWatchlistInput),
            "output": _model_to_schema(app_models.GetWatchlistOutput),
        },
        "SearchStocks": {
            "input": _model_to_schema(app_models.SearchStocksInput),
            "output": _model_to_schema(app_models.SearchStocksOutput),
        },
        "GetStockQuote": {
            "input": _model_to_schema(app_models.GetStockQuoteInput),
            "output": _model_to_schema(app_models.GetStockQuoteOutput),
        },
        "BuyStock": {
            "input": _model_to_schema(app_models.BuyStockInput),
            "output": _model_to_schema(app_models.BuyStockOutput),
        },
        "SellStock": {
            "input": _model_to_schema(app_models.SellStockInput),
            "output": _model_to_schema(app_models.SellStockOutput),
        },
    }
    
    # WhatsApp APIs
    schemas["WhatsApp"] = {
        "GetMessages": {
            "input": _model_to_schema(app_models.GetMessagesInput),
            "output": _model_to_schema(app_models.GetMessagesOutput),
        },
        "SendMessage": {
            "input": _model_to_schema(app_models.SendMessageInput),
            "output": _model_to_schema(app_models.SendMessageOutput),
        },
        "SendMedia": {
            "input": _model_to_schema(app_models.SendMediaInput),
            "output": _model_to_schema(app_models.SendMediaOutput),
        },
    }
    
    # Gmail APIs
    schemas["Gmail"] = {
        "GetInbox": {
            "input": _model_to_schema(app_models.GetInboxInput),
            "output": _model_to_schema(app_models.GetInboxOutput),
        },
        "ReadEmail": {
            "input": _model_to_schema(app_models.ReadEmailInput),
            "output": _model_to_schema(app_models.ReadEmailOutput),
        },
        "SendEmail": {
            "input": _model_to_schema(app_models.SendEmailInput),
            "output": _model_to_schema(app_models.SendEmailOutput),
        },
        "ReplyEmail": {
            "input": _model_to_schema(app_models.ReplyEmailInput),
            "output": _model_to_schema(app_models.ReplyEmailOutput),
        },
    }
    
    # LinkedIn APIs
    schemas["LinkedIn"] = {
        "UpdateProfile": {
            "input": _model_to_schema(app_models.UpdateProfileInput),
            "output": _model_to_schema(app_models.UpdateProfileOutput),
        },
        "AddExperience": {
            "input": _model_to_schema(app_models.AddExperienceInput),
            "output": _model_to_schema(app_models.AddExperienceOutput),
        },
        "AddSkill": {
            "input": _model_to_schema(app_models.AddSkillInput),
            "output": _model_to_schema(app_models.AddSkillOutput),
        },
        "PostUpdate": {
            "input": _model_to_schema(app_models.PostUpdateInput),
            "output": _model_to_schema(app_models.PostUpdateOutput),
        },
        "GetFeed": {
            "input": _model_to_schema(app_models.GetFeedInput),
            "output": _model_to_schema(app_models.GetFeedOutput),
        },
        "LikePost": {
            "input": _model_to_schema(app_models.LikePostInput),
            "output": _model_to_schema(app_models.LikePostOutput),
        },
        "CommentOnPost": {
            "input": _model_to_schema(app_models.CommentOnPostInput),
            "output": _model_to_schema(app_models.CommentOnPostOutput),
        },
        "SearchJobs": {
            "input": _model_to_schema(app_models.SearchJobsInput),
            "output": _model_to_schema(app_models.SearchJobsOutput),
        },
        "ApplyJob": {
            "input": _model_to_schema(app_models.ApplyJobInput),
            "output": _model_to_schema(app_models.ApplyJobOutput),
        },
        "SendConnectionRequest": {
            "input": _model_to_schema(app_models.SendConnectionRequestInput),
            "output": _model_to_schema(app_models.SendConnectionRequestOutput),
        },
    }
    
    # Notion APIs - use generic schemas since some overlap with Netflix
    schemas["Notion"] = {
        "GetPages": {
            "input": _model_to_schema(app_models.GetPagesInput),
            "output": _model_to_schema(app_models.GetPagesOutput),
        },
        "CreatePage": {
            "input": _model_to_schema(app_models.CreatePageInput),
            "output": _model_to_schema(app_models.CreatePageOutput),
        },
        "UpdatePage": {
            "input": _model_to_schema(app_models.UpdatePageInput),
            "output": _model_to_schema(app_models.UpdatePageOutput),
        },
        "SearchContent": {
            "input": _model_to_schema(app_models.NotionSearchContentInput),
            "output": _model_to_schema(app_models.NotionSearchContentOutput),
        },
        "CreateDatabaseEntry": {
            "input": _model_to_schema(app_models.CreateDatabaseEntryInput),
            "output": _model_to_schema(app_models.CreateDatabaseEntryOutput),
        },
    }
    
    # Netflix APIs
    schemas["Netflix"] = {
        "SearchContent": {
            "input": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            "output": {"type": "object", "properties": {"titles": {"type": "array", "items": {"type": "object"}}}, "required": ["titles"]},
        },
        "ShowTitle": {
            "input": _model_to_schema(app_models.ShowTitleInput),
            "output": _model_to_schema(app_models.ShowTitleOutput),
        },
        "PlayContent": {
            "input": _model_to_schema(app_models.PlayContentInput),
            "output": _model_to_schema(app_models.PlayContentOutput),
        },
        "AddToMyList": {
            "input": _model_to_schema(app_models.AddToMyListInput),
            "output": _model_to_schema(app_models.AddToMyListOutput),
        },
        "RateContent": {
            "input": _model_to_schema(app_models.RateContentInput),
            "output": _model_to_schema(app_models.RateContentOutput),
        },
    }
    
    # Goodreads APIs
    schemas["Goodreads"] = {
        "SearchBooks": {
            "input": _model_to_schema(app_models.SearchBooksInput),
            "output": _model_to_schema(app_models.SearchBooksOutput),
        },
        "ShowBook": {
            "input": _model_to_schema(app_models.ShowBookInput),
            "output": _model_to_schema(app_models.ShowBookOutput),
        },
        "AddToShelf": {
            "input": _model_to_schema(app_models.AddToShelfInput),
            "output": _model_to_schema(app_models.AddToShelfOutput),
        },
        "RateBook": {
            "input": _model_to_schema(app_models.RateBookInput),
            "output": _model_to_schema(app_models.RateBookOutput),
        },
        "WriteReview": {
            "input": _model_to_schema(app_models.WriteReviewInput),
            "output": _model_to_schema(app_models.WriteReviewOutput),
        },
    }
    
    # Instagram APIs
    schemas["Instagram"] = {
        "PostStory": {
            "input": _model_to_schema(app_models.PostStoryInput),
            "output": _model_to_schema(app_models.PostStoryOutput),
        },
        "LikePost": {
            "input": {"type": "object", "properties": {"post_id": {"type": "string"}}, "required": ["post_id"]},
            "output": {"type": "object", "properties": {"success": {"type": "boolean"}, "post_id": {"type": "string"}, "new_likes_count": {"type": "integer"}}, "required": ["success", "post_id", "new_likes_count"]},
        },
        "CommentOnPost": {
            "input": {"type": "object", "properties": {"post_id": {"type": "string"}, "comment": {"type": "string"}}, "required": ["post_id", "comment"]},
            "output": {"type": "object", "properties": {"success": {"type": "boolean"}, "post_id": {"type": "string"}, "comment_timestamp": {"type": "string"}}, "required": ["success", "post_id", "comment_timestamp"]},
        },
        "SendDirectMessage": {
            "input": _model_to_schema(app_models.SendDirectMessageInput),
            "output": _model_to_schema(app_models.SendDirectMessageOutput),
        },
        "FollowUser": {
            "input": _model_to_schema(app_models.FollowUserInput),
            "output": _model_to_schema(app_models.FollowUserOutput),
        },
        "UnfollowUser": {
            "input": _model_to_schema(app_models.UnfollowUserInput),
            "output": _model_to_schema(app_models.UnfollowUserOutput),
        },
        "GetFollowing": {
            "input": _model_to_schema(app_models.GetFollowingInput),
            "output": _model_to_schema(app_models.GetFollowingOutput),
        },
    }
    
    # Google APIs
    schemas["Google"] = {
        "Search": {
            "input": _model_to_schema(app_models.GoogleSearchInput),
            "output": _model_to_schema(app_models.GoogleSearchOutput),
        },
        "ClickResult": {
            "input": _model_to_schema(app_models.ClickResultInput),
            "output": _model_to_schema(app_models.ClickResultOutput),
        },
    }
    
    # LLM Assistant APIs
    schemas["LLM Assistant"] = {
        "CreateConversation": {
            "input": _model_to_schema(app_models.CreateConversationInput),
            "output": _model_to_schema(app_models.CreateConversationOutput),
        },
        "ContinueConversation": {
            "input": _model_to_schema(app_models.ContinueConversationInput),
            "output": _model_to_schema(app_models.ContinueConversationOutput),
        },
    }
    
    return schemas


# Build schemas on module load
APP_API_SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = _build_api_schemas()


def get_api_input_output_models(app_name: str, api_name: str):
    """Get the Pydantic Input and Output model classes for a specific API."""
    from mem_bench.behavior_and_conversation import app_models
    
    model_map = {
        ("Amazon", "SearchProducts"): (app_models.SearchProductsInput, app_models.SearchProductsOutput),
        ("Amazon", "ShowProduct"): (app_models.ShowProductInput, app_models.ShowProductOutput),
        ("Amazon", "AddToCart"): (app_models.AddToCartInput, app_models.AddToCartOutput),
        ("Amazon", "ShowCart"): (app_models.ShowCartInput, app_models.ShowCartOutput),
        ("Amazon", "ShowWishlist"): (app_models.ShowWishlistInput, app_models.ShowWishlistOutput),
        ("Amazon", "Checkout"): (app_models.CheckoutInput, app_models.CheckoutOutput),
        ("Spotify", "SearchSongs"): (app_models.SearchSongsInput, app_models.SearchSongsOutput),
        ("Spotify", "PlaySong"): (app_models.PlaySongInput, app_models.PlaySongOutput),
        ("Spotify", "AddToPlaylist"): (app_models.AddToPlaylistInput, app_models.AddToPlaylistOutput),
        ("Spotify", "FollowArtist"): (app_models.FollowArtistInput, app_models.FollowArtistOutput),
        ("Fitbit", "LogWorkout"): (app_models.LogWorkoutInput, app_models.LogWorkoutOutput),
        ("Fitbit", "SyncDevice"): (app_models.SyncDeviceInput, app_models.SyncDeviceOutput),
        ("Fitbit", "SetGoals"): (app_models.SetGoalsInput, app_models.SetGoalsOutput),
        ("Chase", "GetBalance"): (app_models.GetBalanceInput, app_models.GetBalanceOutput),
        ("Chase", "GetTransactions"): (app_models.GetTransactionsInput, app_models.GetTransactionsOutput),
        ("Chase", "SearchTransactions"): (app_models.SearchTransactionsInput, app_models.SearchTransactionsOutput),
        ("Chase", "TransferMoney"): (app_models.TransferMoneyInput, app_models.TransferMoneyOutput),
        ("Chase", "PayBill"): (app_models.PayBillInput, app_models.PayBillOutput),
        ("Robinhood", "GetPortfolio"): (app_models.GetPortfolioInput, app_models.GetPortfolioOutput),
        ("Robinhood", "GetWatchlist"): (app_models.GetWatchlistInput, app_models.GetWatchlistOutput),
        ("Robinhood", "SearchStocks"): (app_models.SearchStocksInput, app_models.SearchStocksOutput),
        ("Robinhood", "GetStockQuote"): (app_models.GetStockQuoteInput, app_models.GetStockQuoteOutput),
        ("Robinhood", "BuyStock"): (app_models.BuyStockInput, app_models.BuyStockOutput),
        ("Robinhood", "SellStock"): (app_models.SellStockInput, app_models.SellStockOutput),
        ("WhatsApp", "GetMessages"): (app_models.GetMessagesInput, app_models.GetMessagesOutput),
        ("WhatsApp", "SendMessage"): (app_models.SendMessageInput, app_models.SendMessageOutput),
        ("WhatsApp", "SendMedia"): (app_models.SendMediaInput, app_models.SendMediaOutput),
        ("Gmail", "GetInbox"): (app_models.GetInboxInput, app_models.GetInboxOutput),
        ("Gmail", "ReadEmail"): (app_models.ReadEmailInput, app_models.ReadEmailOutput),
        ("Gmail", "SendEmail"): (app_models.SendEmailInput, app_models.SendEmailOutput),
        ("Gmail", "ReplyEmail"): (app_models.ReplyEmailInput, app_models.ReplyEmailOutput),
        ("LinkedIn", "UpdateProfile"): (app_models.UpdateProfileInput, app_models.UpdateProfileOutput),
        ("LinkedIn", "AddExperience"): (app_models.AddExperienceInput, app_models.AddExperienceOutput),
        ("LinkedIn", "AddSkill"): (app_models.AddSkillInput, app_models.AddSkillOutput),
        ("LinkedIn", "PostUpdate"): (app_models.PostUpdateInput, app_models.PostUpdateOutput),
        ("LinkedIn", "GetFeed"): (app_models.GetFeedInput, app_models.GetFeedOutput),
        ("LinkedIn", "LikePost"): (app_models.LikePostInput, app_models.LikePostOutput),
        ("LinkedIn", "CommentOnPost"): (app_models.CommentOnPostInput, app_models.CommentOnPostOutput),
        ("LinkedIn", "SearchJobs"): (app_models.SearchJobsInput, app_models.SearchJobsOutput),
        ("LinkedIn", "ApplyJob"): (app_models.ApplyJobInput, app_models.ApplyJobOutput),
        ("LinkedIn", "SendConnectionRequest"): (app_models.SendConnectionRequestInput, app_models.SendConnectionRequestOutput),
        ("Notion", "GetPages"): (app_models.GetPagesInput, app_models.GetPagesOutput),
        ("Notion", "CreatePage"): (app_models.CreatePageInput, app_models.CreatePageOutput),
        ("Notion", "UpdatePage"): (app_models.UpdatePageInput, app_models.UpdatePageOutput),
        ("Notion", "SearchContent"): (app_models.NotionSearchContentInput, app_models.NotionSearchContentOutput),
        ("Notion", "CreateDatabaseEntry"): (app_models.CreateDatabaseEntryInput, app_models.CreateDatabaseEntryOutput),
        ("Netflix", "ShowTitle"): (app_models.ShowTitleInput, app_models.ShowTitleOutput),
        ("Netflix", "PlayContent"): (app_models.PlayContentInput, app_models.PlayContentOutput),
        ("Netflix", "AddToMyList"): (app_models.AddToMyListInput, app_models.AddToMyListOutput),
        ("Netflix", "RateContent"): (app_models.RateContentInput, app_models.RateContentOutput),
        ("Goodreads", "SearchBooks"): (app_models.SearchBooksInput, app_models.SearchBooksOutput),
        ("Goodreads", "ShowBook"): (app_models.ShowBookInput, app_models.ShowBookOutput),
        ("Goodreads", "AddToShelf"): (app_models.AddToShelfInput, app_models.AddToShelfOutput),
        ("Goodreads", "RateBook"): (app_models.RateBookInput, app_models.RateBookOutput),
        ("Goodreads", "WriteReview"): (app_models.WriteReviewInput, app_models.WriteReviewOutput),
        ("Instagram", "PostStory"): (app_models.PostStoryInput, app_models.PostStoryOutput),
        ("Instagram", "SendDirectMessage"): (app_models.SendDirectMessageInput, app_models.SendDirectMessageOutput),
        ("Instagram", "FollowUser"): (app_models.FollowUserInput, app_models.FollowUserOutput),
        ("Instagram", "UnfollowUser"): (app_models.UnfollowUserInput, app_models.UnfollowUserOutput),
        ("Instagram", "GetFollowing"): (app_models.GetFollowingInput, app_models.GetFollowingOutput),
        ("Google", "Search"): (app_models.GoogleSearchInput, app_models.GoogleSearchOutput),
        ("Google", "ClickResult"): (app_models.ClickResultInput, app_models.ClickResultOutput),
        ("LLM Assistant", "CreateConversation"): (app_models.CreateConversationInput, app_models.CreateConversationOutput),
        ("LLM Assistant", "ContinueConversation"): (app_models.ContinueConversationInput, app_models.ContinueConversationOutput),
    }
    
    return model_map.get((app_name, api_name), (None, None))


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

    def _append_to_history(self, key: str, item: Any, max_length: int = MAX_HISTORY_LENGTH) -> None:
        """
        Append an item to a history list in state, keeping only the most recent entries.
        
        Args:
            key: The state key for the history list
            item: The item to append
            max_length: Maximum number of entries to keep (default: MAX_HISTORY_LENGTH)
        """
        history = self.state.setdefault(key, [])
        history.append(item)
        if len(history) > max_length:
            self.state[key] = history[-max_length:]

    def _trim_history(self, key: str, max_length: int = MAX_HISTORY_LENGTH) -> None:
        """
        Trim a history list in state to the most recent entries.
        
        Args:
            key: The state key for the history list  
            max_length: Maximum number of entries to keep (default: MAX_HISTORY_LENGTH)
        """
        if key in self.state and isinstance(self.state[key], list):
            if len(self.state[key]) > max_length:
                self.state[key] = self.state[key][-max_length:]

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
    """Amazon e-commerce app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Amazon", user_id)
        self._product_cache: Dict[str, Dict] = {}  # Cache products for consistency

    def _initialize_state(self) -> None:
        """Initialize Amazon state based on AmazonState model."""
        self.state = {
            "user_id": self.user_id,
            "prime_member": random.choice([True, False]),
            "order_history": [],
            "search_history": [],
            "viewed_products": [],
            "cart": [],
            "wishlist": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Amazon API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "SearchProducts":
            return self._search_products(timestamp, description, context)
        elif api_name == "ShowProduct":
            return self._show_product(timestamp, description, context)
        elif api_name == "AddToCart":
            return self._add_to_cart(timestamp, description, context)
        elif api_name == "ShowCart":
            return self._show_cart(timestamp, description, context)
        elif api_name == "ShowWishlist":
            return self._show_wishlist(timestamp, description, context)
        elif api_name == "Checkout":
            return self._checkout(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Amazon API: {api_name}")

    def _search_products(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search for products on Amazon using keywords."""
        query = self._extract_search_query(description)
        products = self._generate_search_results(query, context)

        # Update search history (bounded)
        self._append_to_history("search_history", query)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchProducts",
            request={"query": query},
            response={
                "products": products,
                "search_timestamp": timestamp
            }
        )

    def _show_product(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View detailed information about a specific product."""
        product_id = self._extract_product_id(description)
        product = self._get_or_create_product(product_id, description, context)
        reviews = self._generate_reviews(product["name"], product["rating"])

        # Track viewed product (store full product object, bounded)
        viewed_ids = [p.get("product_id") for p in self.state["viewed_products"]]
        if product_id not in viewed_ids:
            self._append_to_history("viewed_products", product)

        in_cart = any(item.get("product_id") == product_id for item in self.state["cart"])
        wishlist_ids = [p.get("product_id") for p in self.state["wishlist"]]
        in_wishlist = product_id in wishlist_ids

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowProduct",
            request={"product_id": product_id},
            response={
                "product": product,
                "reviews": reviews,
                "in_cart": in_cart,
                "in_wishlist": in_wishlist
            }
        )

    def _add_to_cart(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Add a product to the shopping cart."""
        product_id = self._extract_product_id(description)
        quantity = self._extract_quantity(description)
        product = self._get_or_create_product(product_id, description, context)

        # Check if product already in cart
        for item in self.state["cart"]:
            if item.get("product_id") == product_id:
                item["quantity"] += quantity
                break
        else:
            self.state["cart"].append({
                "product_id": product_id,
                "name": product["name"],
                "price": product["price"],
                "quantity": quantity
            })

        cart_total = sum(item["price"] * item["quantity"] for item in self.state["cart"])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="AddToCart",
            request={"product_id": product_id, "quantity": quantity},
            response={
                "cart": self.state["cart"],
                "cart_total": round(cart_total, 2)
            }
        )

    def _show_cart(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View all items currently in the shopping cart."""
        cart_items = []
        for item in self.state["cart"]:
            cart_items.append({
                "product_id": item.get("product_id"),
                "name": item.get("name"),
                "price": item.get("price"),
                "quantity": item.get("quantity")
            })

        cart_total = sum(item["price"] * item["quantity"] for item in self.state["cart"])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowCart",
            request={},
            response={
                "cart_items": cart_items,
                "cart_total": round(cart_total, 2),
                "prime_member": self.state["prime_member"]
            }
        )

    def _show_wishlist(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View all items saved in the wishlist."""
        wishlist_items = []
        for product in self.state["wishlist"]:
            wishlist_items.append({
                "product_id": product.get("product_id"),
                "name": product.get("name"),
                "price": product.get("price"),
                "added_at": timestamp  # Now using datetime format
            })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowWishlist",
            request={},
            response={
                "wishlist_items": wishlist_items
            }
        )

    def _checkout(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Complete the purchase of items in the cart."""
        if not self.state["cart"]:
            # If cart is empty, create a single item from description
            product_id = self._extract_product_id(description)
            product = self._get_or_create_product(product_id, description, context)
            self.state["cart"].append({
                "product_id": product_id,
                "name": product["name"],
                "price": product["price"],
                "quantity": 1
            })

        order_items = []
        total_price = 0.0
        for item in self.state["cart"]:
            order_items.append({
                "product_id": item.get("product_id"),
                "name": item.get("name"),
                "price": item.get("price"),
                "quantity": item.get("quantity")
            })
            total_price += item["price"] * item["quantity"]

        order_id = f"AMZ-{uuid.uuid4().hex[:8].upper()}"
        order_date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Prime members get faster delivery
        delivery_days = random.randint(1, 2) if self.state["prime_member"] else random.randint(3, 5)
        estimated_delivery = (order_date + timedelta(days=delivery_days)).strftime("%Y-%m-%d")

        # Record order in history (bounded)
        order = {
            "order_id": order_id,
            "items": order_items,
            "total_price": round(total_price, 2),
            "order_date": timestamp
        }
        self._append_to_history("order_history", order)

        # Clear cart
        self.state["cart"] = []

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Checkout",
            request={},
            response={
                "order_id": order_id,
                "order_items": order_items,
                "total_price": round(total_price, 2),
                "order_date": timestamp,
                "estimated_delivery": estimated_delivery,
                "prime_member": self.state["prime_member"]
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

    def _extract_product_id(self, description: str) -> str:
        """Extract or generate product ID from description."""
        import re
        # Try to find existing product ID
        match = re.search(r'PROD[A-Z0-9]+', description)
        if match:
            return match.group(0)
        # Generate from description
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            name = quoted[0][0] or quoted[0][1]
            return f"PROD{abs(hash(name)) % 100000:05d}"
        return f"PROD{random.randint(10000, 99999)}"

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
            product_id = f"PROD{abs(hash(query + str(i))) % 100000:05d}"
            product = {
                "product_id": product_id,
                "name": f"{query.title()} - Option {chr(65+i)}",
                "price": round(random.uniform(19.99, 299.99), 2),
                "category": self._infer_category(query),
                "rating": round(random.uniform(3.5, 5.0), 1)
            }
            products.append(product)
            self._product_cache[product_id] = product

        return products

    def _get_or_create_product(self, product_id: str, description: str, context: Dict[str, Any]) -> Dict:
        """Get or create product by ID."""
        if product_id in self._product_cache:
            return self._product_cache[product_id]

        # Extract name from description
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        name = quoted[0][0] or quoted[0][1] if quoted else f"Product {product_id}"

        product = {
            "product_id": product_id,
            "name": name,
            "price": round(random.uniform(19.99, 299.99), 2),
            "category": self._infer_category(name),
            "rating": round(random.uniform(3.5, 5.0), 1)
        }
        self._product_cache[product_id] = product
        return product

    def _infer_category(self, text: str) -> str:
        """Infer product category from text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["phone", "laptop", "computer", "tablet", "headphone"]):
            return "Electronics"
        elif any(w in text_lower for w in ["shirt", "pants", "dress", "shoes", "jacket"]):
            return "Clothing"
        elif any(w in text_lower for w in ["book", "novel", "textbook"]):
            return "Books"
        elif any(w in text_lower for w in ["kitchen", "cookware", "appliance"]):
            return "Home & Kitchen"
        return "General"

    def _generate_reviews(self, product_name: str, rating: float) -> List[Dict[str, Any]]:
        """Generate review samples."""
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
                "rating": review_rating,
                "text": random.choice(review_templates),
                "author": random.choice(["Alex", "Jamie", "Taylor", "Morgan", "Riley"])
            })
        return reviews


class GoogleApp(BaseApp):
    """Google search app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Google", user_id)
        self._results_cache: Dict[str, List[Dict]] = {}  # query -> results

    def _initialize_state(self) -> None:
        """Initialize Google state based on GoogleState model."""
        self.state = {
            "user_id": self.user_id,
            "search_history": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Google API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "Search":
            return self._search(timestamp, description, context)
        elif api_name == "ClickResult":
            return self._click_result(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Google API: {api_name}")

    def _search(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Perform a web search and receive list of results."""
        query = self._extract_search_query(description)
        results = self._generate_search_results(query)

        # Store in cache for ClickResult
        self._results_cache[query] = results

        # Record search history (bounded)
        search_record = {
            "query": query,
            "results": results,
            "searched_at": timestamp,
            "clicked_result_id": None
        }
        self._append_to_history("search_history", search_record)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Search",
            request={"query": query},
            response={
                "results": results,
                "search_timestamp": timestamp
            }
        )

    def _click_result(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Click on a specific search result to view the webpage."""
        result_id = self._extract_result_id(description)
        search_query = self._extract_search_query(description)

        # Find the result
        result = None
        if search_query in self._results_cache:
            for r in self._results_cache[search_query]:
                if r.get("result_id") == result_id:
                    result = r
                    break

        # If not found, create a placeholder
        if not result:
            result = {
                "result_id": result_id,
                "title": f"Search Result {result_id}",
                "snippet": "Detailed information about the search topic."
            }

        # Update search history with clicked result
        for record in reversed(self.state["search_history"]):
            if record.get("query") == search_query:
                record["clicked_result_id"] = result_id
                break

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ClickResult",
            request={"result_id": result_id, "search_query": search_query},
            response={
                "result": result,
                "clicked_at": timestamp
            }
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:6])

    def _extract_result_id(self, description: str) -> str:
        """Extract result ID from description."""
        import re
        match = re.search(r'RES[A-Z0-9]+', description)
        if match:
            return match.group(0)
        return f"RES{random.randint(1000, 9999)}"

    def _generate_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate search results."""
        results = []
        sources = ["Wikipedia", "News article", "Academic paper", "Blog post", "Forum discussion", "Official site"]
        for idx in range(random.randint(3, 6)):
            result_id = f"RES{abs(hash(query + str(idx))) % 10000:04d}"
            results.append({
                "result_id": result_id,
                "title": f"{query.title()} - Result {idx + 1}",
                "snippet": f"Relevant information about {query}. This result provides useful context and details.",
                "source": random.choice(sources)
            })
        return results


class SpotifyApp(BaseApp):
    """Spotify music streaming app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Spotify", user_id)
        self._song_cache: Dict[str, Dict] = {}

    def _initialize_state(self) -> None:
        """Initialize Spotify state based on SpotifyState model."""
        self.state = {
            "user_id": self.user_id,
            "premium": random.choice([True, False]),
            "playlists": self._create_initial_playlists(),
            "followed_artists": [],
            "play_history": [],
            "favorite_genres": random.sample(["Pop", "Rock", "Hip Hop", "Electronic", "R&B", "Jazz", "Classical"], 3)
        }

    def _create_initial_playlists(self) -> List[Dict]:
        """Create some initial playlists."""
        playlists = []
        playlist_names = ["My Favorites", "Workout Mix", "Chill Vibes"]
        for name in playlist_names:
            playlist_id = f"PL{uuid.uuid4().hex[:8].upper()}"
            playlists.append({
                "playlist_id": playlist_id,
                "name": name,
                "song_ids": [SpotifyDatabase.get_random_song().get("song_id", f"SONG{random.randint(1000,9999)}") for _ in range(3)]
            })
        return playlists

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Spotify API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "SearchSongs":
            return self._search_songs(timestamp, description, context)
        elif api_name == "PlaySong":
            return self._play_song(timestamp, description, context)
        elif api_name == "AddToPlaylist":
            return self._add_to_playlist(timestamp, description, context)
        elif api_name == "FollowArtist":
            return self._follow_artist(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Spotify API: {api_name}")

    def _search_songs(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search for songs, artists, or albums."""
        query = self._extract_search_query(description)
        songs = self._generate_song_results(query)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchSongs",
            request={"query": query},
            response={
                "songs": songs,
                "search_timestamp": timestamp
            }
        )

    def _play_song(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Play a specific song and track listening duration."""
        song_id = self._extract_song_id(description)
        song = self._get_or_create_song(song_id, description, context)

        # Record play history (bounded)
        play_record = {
            "song_id": song_id,
            "played_at": timestamp,
            "duration_played_minutes": song.get("duration_minutes", random.randint(3, 5))
        }
        self._append_to_history("play_history", play_record)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="PlaySong",
            request={"song_id": song_id},
            response={
                "song": song,
                "playing_status": PlayingStatus.PLAYING.value,
                "premium": self.state["premium"],
                "play_started_at": timestamp
            }
        )

    def _add_to_playlist(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Add a song to a specific playlist."""
        playlist_id, playlist_name = self._extract_playlist_info(description)
        song_id = self._extract_song_id(description)

        # Find or create playlist
        playlist = None
        for pl in self.state["playlists"]:
            if pl["playlist_id"] == playlist_id or pl["name"].lower() == playlist_name.lower():
                playlist = pl
                break

        if not playlist:
            playlist = {
                "playlist_id": playlist_id or f"PL{uuid.uuid4().hex[:8].upper()}",
                "name": playlist_name,
                "song_ids": []
            }
            self.state["playlists"].append(playlist)

        # Add song to playlist
        if song_id not in playlist["song_ids"]:
            playlist["song_ids"].append(song_id)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="AddToPlaylist",
            request={"playlist_id": playlist["playlist_id"], "song_id": song_id},
            response={
                "playlist": playlist
            }
        )

    def _follow_artist(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Follow an artist to receive updates and recommendations."""
        artist_id, artist_name = self._extract_artist_info(description)

        if artist_id not in self.state["followed_artists"]:
            self._append_to_history("followed_artists", artist_id)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="FollowArtist",
            request={"artist_id": artist_id, "artist_name": artist_name},
            response={
                "followed_artists": self.state["followed_artists"]
            }
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "popular music"

    def _extract_song_id(self, description: str) -> str:
        """Extract or generate song ID from description."""
        import re
        match = re.search(r'SONG[A-Z0-9]+', description)
        if match:
            return match.group(0)
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            name = quoted[0][0] or quoted[0][1]
            return f"SONG{abs(hash(name)) % 100000:05d}"
        return f"SONG{random.randint(10000, 99999)}"

    def _extract_playlist_info(self, description: str) -> Tuple[str, str]:
        """Extract playlist ID and name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        playlist_name = quoted[0][0] or quoted[0][1] if quoted else "My Playlist"
        
        match = re.search(r'PL[A-Z0-9]+', description)
        playlist_id = match.group(0) if match else f"PL{abs(hash(playlist_name)) % 100000:05d}"
        
        return playlist_id, playlist_name

    def _extract_artist_info(self, description: str) -> Tuple[str, str]:
        """Extract artist ID and name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        artist_name = quoted[0][0] or quoted[0][1] if quoted else "Unknown Artist"
        
        # Look for "by" pattern
        match = re.search(r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', description)
        if match:
            artist_name = match.group(1).strip()
        
        artist_id = f"ART{abs(hash(artist_name)) % 100000:05d}"
        return artist_id, artist_name

    def _generate_song_results(self, query: str) -> List[Dict]:
        """Generate song search results."""
        songs = SpotifyDatabase.search_songs(query, limit=8)
        # Ensure songs have the required fields
        for song in songs:
            if "song_id" not in song:
                song["song_id"] = f"SONG{random.randint(10000, 99999)}"
            if "duration_minutes" not in song:
                song["duration_minutes"] = random.randint(3, 5)
            self._song_cache[song["song_id"]] = song
        return songs

    def _get_or_create_song(self, song_id: str, description: str, context: Dict[str, Any]) -> Dict:
        """Get or create a song by ID."""
        if song_id in self._song_cache:
            return self._song_cache[song_id]

        # Try to get from database
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        
        if quoted:
            song_info = quoted[0][0] or quoted[0][1]
            # Try to split by 'by' or '-'
            if " by " in song_info.lower():
                parts = song_info.split(" by ", 1)
                title = parts[0].strip()
                artist = parts[1].strip() if len(parts) > 1 else "Unknown Artist"
            elif " - " in song_info:
                parts = song_info.split(" - ", 1)
                title = parts[0].strip()
                artist = parts[1].strip() if len(parts) > 1 else "Unknown Artist"
        else:
                title = song_info
                artist = "Unknown Artist"

        song = {
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "genre": random.choice(self.state.get("favorite_genres", ["Pop"])),
            "duration_minutes": random.randint(3, 5)
        }
        self._song_cache[song_id] = song
        return song


class FitbitApp(BaseApp):
    """Fitbit health & fitness tracking app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Fitbit", user_id)

    def _initialize_state(self) -> None:
        """Initialize Fitbit state based on FitbitState model."""
        self.state = {
            "user_id": self.user_id,
            "goals": [
                {"goal_type": "steps", "target_value": 10000},
                {"goal_type": "active_minutes", "target_value": 30},
            ],
            "workout_history": [],
            "daily_syncs": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Fitbit API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "LogWorkout":
            return self._log_workout(timestamp, description, context)
        elif api_name == "SyncDevice":
            return self._sync_device(timestamp, description, context)
        elif api_name == "SetGoals":
            return self._set_goals(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Fitbit API: {api_name}")

    def _log_workout(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Manually log a workout session with type, duration, and intensity."""
        activity_type = self._extract_activity_type(description)
        duration_minutes = self._extract_duration(description)
        intensity = self._extract_intensity(description)
        calories_burned = self._calculate_calories(activity_type, duration_minutes, intensity)

        workout_id = f"WKT{uuid.uuid4().hex[:8].upper()}"
        workout = {
            "workout_id": workout_id,
            "activity_type": activity_type,
            "duration_minutes": duration_minutes,
            "intensity": intensity,
            "calories_burned": calories_burned,
            "timestamp": timestamp
        }
        self._append_to_history("workout_history", workout)

        # Calculate today's total active minutes
        date = timestamp.split()[0]
        today_active_minutes = sum(
            w["duration_minutes"] for w in self.state["workout_history"]
            if w["timestamp"].startswith(date)
        )

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="LogWorkout",
            request={
                "activity_type": activity_type,
                "duration_minutes": duration_minutes,
                "intensity": intensity
            },
            response={
                "workout": workout,
                "calories_burned": calories_burned,
                "today_total_active_minutes": today_active_minutes
            }
        )

    def _sync_device(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Sync wearable device data including steps, heart rate, sleep patterns."""
        date = timestamp.split()[0]

        # Generate sync data
        sync_data = {
            "sync_date": date,
            "steps": random.randint(3000, 15000),
            "active_minutes": random.randint(15, 90),
            "calories_burned": random.randint(1500, 2500),
            "sleep_hours": round(random.uniform(5.5, 9.0), 1),
            "avg_heart_rate": random.randint(60, 85)
        }

        # Store sync data (bounded)
        self._append_to_history("daily_syncs", sync_data)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SyncDevice",
            request={},
            response={
                "sync_data": sync_data,
                "sync_timestamp": timestamp
            }
        )

    def _set_goals(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Set or update fitness goals such as daily steps, active minutes, or weight targets."""
        goals = self._extract_goals(description)

        # Update existing goals or add new ones
        for new_goal in goals:
            found = False
            for existing_goal in self.state["goals"]:
                if existing_goal["goal_type"] == new_goal["goal_type"]:
                    existing_goal["target_value"] = new_goal["target_value"]
                    found = True
                break
            if not found:
                self.state["goals"].append(new_goal)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SetGoals",
            request={"goals": goals},
            response={
                "goals": self.state["goals"],
                "updated_at": timestamp
            }
        )

    def _extract_activity_type(self, description: str) -> str:
        """Extract activity type from description."""
        desc_lower = description.lower()
        if "run" in desc_lower or "jog" in desc_lower:
            return "running"
        elif "bike" in desc_lower or "cycl" in desc_lower:
            return "cycling"
        elif "swim" in desc_lower:
            return "swimming"
        elif "walk" in desc_lower:
            return "walking"
        elif "yoga" in desc_lower:
            return "yoga"
        elif "weight" in desc_lower or "strength" in desc_lower:
            return "strength_training"
        return random.choice(["running", "cycling", "walking", "strength_training"])

    def _extract_duration(self, description: str) -> int:
        """Extract duration in minutes from description."""
        import re
        match = re.search(r'(\d+)\s*(?:min|minute)', description, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)\s*(?:hour|hr)', description, re.IGNORECASE)
        if match:
            return int(match.group(1)) * 60
        return random.randint(20, 60)

    def _extract_intensity(self, description: str) -> str:
        """Extract intensity from description."""
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["intense", "hard", "vigorous", "high"]):
            return "high"
        elif any(w in desc_lower for w in ["easy", "light", "gentle", "low"]):
            return "low"
        return "medium"

    def _extract_goals(self, description: str) -> List[Dict]:
        """Extract goals from description."""
        import re
        goals = []
        desc_lower = description.lower()

        # Look for steps goal
        match = re.search(r'(\d+)\s*steps', desc_lower)
        if match:
            goals.append({"goal_type": "steps", "target_value": int(match.group(1))})

        # Look for active minutes goal
        match = re.search(r'(\d+)\s*(?:active\s*)?minutes?', desc_lower)
        if match:
            goals.append({"goal_type": "active_minutes", "target_value": int(match.group(1))})

        # Look for weight goal
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lbs?|kg|pounds?)', desc_lower)
        if match:
            goals.append({"goal_type": "weight", "target_value": float(match.group(1))})

        if not goals:
            goals = [{"goal_type": "steps", "target_value": 10000}]

        return goals

    def _calculate_calories(self, activity_type: str, duration: int, intensity: str) -> int:
        """Calculate calories burned."""
        base_rates = {
            "running": 10,
            "cycling": 8,
            "swimming": 9,
            "walking": 4,
            "yoga": 3,
            "strength_training": 6
        }
        base_rate = base_rates.get(activity_type, 5)
        intensity_mult = {"low": 0.7, "medium": 1.0, "high": 1.3}.get(intensity, 1.0)
        return int(base_rate * duration * intensity_mult)


class WhatsAppApp(BaseApp):
    """WhatsApp instant messaging app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("WhatsApp", user_id)

    def _initialize_state(self) -> None:
        """Initialize WhatsApp state based on WhatsAppState model."""
        self.state = {
            "user_id": self.user_id,
            "contacts": ["Alice", "Bob", "Carol", "Dave", "Emma", "Family Group", "Work Team"],
            "message_history": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call WhatsApp API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "GetMessages":
            return self._get_messages(timestamp, description, context)
        elif api_name == "SendMessage":
            return self._send_message(timestamp, description, context)
        elif api_name == "SendMedia":
            return self._send_media(timestamp, description, context)
        else:
            raise ValueError(f"Unknown WhatsApp API: {api_name}")

    def _get_messages(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Retrieve message history from a specific contact or group."""
        contact_id = self._extract_contact_id(description)
        limit = self._extract_limit(description)

        # Filter messages for this contact
        messages = [
            msg for msg in self.state["message_history"]
            if msg.get("to_user") == contact_id or msg.get("from_user") == contact_id
        ][-limit:]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetMessages",
            request={"contact_id": contact_id, "limit": limit},
            response={
                "contact_id": contact_id,
                "messages": messages
            }
        )

    def _send_message(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Send a text message to a contact or group."""
        to_user = self._extract_contact_id(description)
        message_content = self._extract_message_content(description)

        message_id = f"MSG{uuid.uuid4().hex[:8].upper()}"
        message = {
            "message_id": message_id,
            "from_user": self.user_id,
            "to_user": to_user,
            "message_type": MessageType.TEXT.value,
            "content": message_content,
            "timestamp": timestamp
        }
        self._append_to_history("message_history", message)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SendMessage",
            request={"to": to_user, "message": message_content},
            response={
                "message": message,
                "sent_timestamp": timestamp
            }
        )

    def _send_media(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Send photos, videos, or voice messages."""
        to_user = self._extract_contact_id(description)
        media_type = self._extract_media_type(description)
        caption = self._extract_caption(description)

        message_id = f"MSG{uuid.uuid4().hex[:8].upper()}"
        message = {
            "message_id": message_id,
            "from_user": self.user_id,
            "to_user": to_user,
            "message_type": MessageType.MEDIA.value,
            "content": f"[{media_type}]" + (f": {caption}" if caption else ""),
            "timestamp": timestamp
        }
        self._append_to_history("message_history", message)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SendMedia",
            request={"to": to_user, "media_type": media_type, "caption": caption},
            response={
                "message": message,
                "sent_timestamp": timestamp
            }
        )

    def _extract_contact_id(self, description: str) -> str:
        """Extract contact ID from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        
        # Look for "to X" pattern
        match = re.search(r'\bto\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)', description)
        if match:
            return match.group(1)
        
        return random.choice(self.state["contacts"])

    def _extract_message_content(self, description: str) -> str:
        """Extract message content from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return description[:100]

    def _extract_media_type(self, description: str) -> str:
        """Extract media type from description."""
        desc_lower = description.lower()
        if "photo" in desc_lower or "image" in desc_lower or "picture" in desc_lower:
            return "photo"
        elif "video" in desc_lower:
            return "video"
        elif "voice" in desc_lower or "audio" in desc_lower:
            return "voice"
        return "photo"

    def _extract_caption(self, description: str) -> Optional[str]:
        """Extract caption from description."""
        import re
        # Look for text after media type keywords
        match = re.search(r'(?:caption|with|saying)\s*[:\s]?\s*["\']?([^"\']+)["\']?', description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_limit(self, description: str) -> int:
        """Extract message limit from description."""
        import re
        match = re.search(r'(\d+)\s*messages?', description, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 50


class ChaseApp(BaseApp):
    """Chase banking & financial management app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Chase", user_id)

    def _initialize_state(self) -> None:
        """Initialize Chase state based on ChaseState model."""
        self.state = {
            "user_id": self.user_id,
            "accounts": [
                {"account_id": "CHK001", "account_type": "checking", "balance": round(random.uniform(1000, 8000), 2)},
                {"account_id": "SAV001", "account_type": "savings", "balance": round(random.uniform(5000, 25000), 2)},
                {"account_id": "CC001", "account_type": "credit_card", "balance": round(-random.uniform(200, 2000), 2)}
            ],
            "transaction_history": self._generate_initial_transactions()
        }

    def _generate_initial_transactions(self) -> List[Dict]:
        """Generate some initial transaction history."""
        merchants = ["Amazon", "Whole Foods", "Starbucks", "Shell Gas", "Netflix", "Uber", "Target"]
        categories = [
            ChaseCategory.SHOPPING.value, ChaseCategory.GROCERIES.value, ChaseCategory.DINING.value,
            ChaseCategory.TRANSPORTATION.value, ChaseCategory.ENTERTAINMENT.value,
            ChaseCategory.TRANSPORTATION.value, ChaseCategory.SHOPPING.value
        ]
        transactions = []
        base_date = datetime.now()
        
        for i in range(10):
            date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            transactions.append({
                "transaction_id": f"TXN{uuid.uuid4().hex[:8].upper()}",
                "transaction_date": date,
                "merchant": merchants[i % len(merchants)],
                "amount": round(random.uniform(5, 150), 2),
                "transaction_type": ChaseTransactionType.DEBIT.value,
                "category": categories[i % len(categories)]
            })
        return transactions

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Chase API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "GetBalance":
            return self._get_balance(timestamp, description, context)
        elif api_name == "GetTransactions":
            return self._get_transactions(timestamp, description, context)
        elif api_name == "SearchTransactions":
            return self._search_transactions(timestamp, description, context)
        elif api_name == "TransferMoney":
            return self._transfer_money(timestamp, description, context)
        elif api_name == "PayBill":
            return self._pay_bill(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Chase API: {api_name}")

    def _get_balance(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Check current account balance."""
        account_id = self._extract_account_id(description)

        if account_id:
            accounts = [acc for acc in self.state["accounts"] if acc["account_id"] == account_id]
        else:
            accounts = self.state["accounts"]

        total_balance = sum(acc["balance"] for acc in accounts)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetBalance",
            request={"account_id": account_id},
            response={
                "accounts": accounts,
                "total_balance": round(total_balance, 2),
                "last_updated": timestamp
            }
        )

    def _get_transactions(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View recent transaction history."""
        account_id = self._extract_account_id(description)
        limit = self._extract_limit(description)

        transactions = self.state["transaction_history"][-limit:]
        
        # Get account balance
        account_balance = 0.0
        for acc in self.state["accounts"]:
            if not account_id or acc["account_id"] == account_id:
                account_balance += acc["balance"]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetTransactions",
            request={"account_id": account_id, "limit": limit},
            response={
                "transactions": transactions,
                "account_balance": round(account_balance, 2)
            }
        )

    def _search_transactions(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search for specific transactions by merchant, amount, or date range."""
        query = self._extract_search_query(description)
        query_lower = query.lower()

        # Search by merchant or category
        results = [
            txn for txn in self.state["transaction_history"]
            if query_lower in txn.get("merchant", "").lower()
            or query_lower in txn.get("category", "").lower()
        ]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchTransactions",
            request={"query": query},
            response={
                "transactions": results
            }
        )

    def _transfer_money(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Transfer funds between accounts or to other people."""
        from_account_id, to_account_id = self._extract_transfer_accounts(description)
        amount = self._extract_amount(description)

        # Find accounts
        from_account = None
        to_account = None
        for acc in self.state["accounts"]:
            if acc["account_id"] == from_account_id:
                from_account = acc
            if acc["account_id"] == to_account_id:
                to_account = acc

        # Perform transfer
        if from_account:
            from_account["balance"] -= amount
        if to_account:
            to_account["balance"] += amount

        transaction_id = f"TRF{uuid.uuid4().hex[:8].upper()}"

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="TransferMoney",
            request={
                "from_account_id": from_account_id,
                "to_account_id": to_account_id,
                "amount": amount
            },
            response={
                "transaction_id": transaction_id,
                "from_account_new_balance": round(from_account["balance"], 2) if from_account else 0,
                "to_account_new_balance": round(to_account["balance"], 2) if to_account else 0,
                "timestamp": timestamp
            }
        )

    def _pay_bill(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Pay bills such as utilities, credit cards, or subscriptions."""
        biller_name = self._extract_biller_name(description)
        amount = self._extract_amount(description)
        from_account_id = self._extract_account_id(description) or "CHK001"

        # Find account and deduct
        account = None
        for acc in self.state["accounts"]:
            if acc["account_id"] == from_account_id:
                account = acc
                break

        if account:
            account["balance"] -= amount

        # Record transaction (bounded)
        transaction = {
            "transaction_id": f"BILL{uuid.uuid4().hex[:8].upper()}",
            "transaction_date": timestamp.split()[0],
            "merchant": biller_name,
            "amount": amount,
            "transaction_type": ChaseTransactionType.DEBIT.value,
            "category": ChaseCategory.BILLS.value
        }
        self._append_to_history("transaction_history", transaction)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="PayBill",
            request={
                "biller_name": biller_name,
                "amount": amount,
                "from_account_id": from_account_id
            },
            response={
                "transaction_id": transaction["transaction_id"],
                "new_balance": round(account["balance"], 2) if account else 0,
                "timestamp": timestamp
            }
        )

    def _extract_account_id(self, description: str) -> Optional[str]:
        """Extract account ID from description."""
        import re
        match = re.search(r'(CHK|SAV|CC)\d+', description, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        
        desc_lower = description.lower()
        if "checking" in desc_lower:
            return "CHK001"
        elif "saving" in desc_lower:
            return "SAV001"
        elif "credit" in desc_lower:
            return "CC001"
        return None

    def _extract_amount(self, description: str) -> float:
        """Extract monetary amount from description."""
        import re
        match = re.search(r'\$?(\d+(?:\.\d{2})?)', description)
        if match:
            return float(match.group(1))
        return round(random.uniform(50, 500), 2)

    def _extract_limit(self, description: str) -> int:
        """Extract limit from description."""
        import re
        match = re.search(r'(\d+)\s*(?:transactions?|items?)', description, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 50

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "transaction"

    def _extract_transfer_accounts(self, description: str) -> Tuple[str, str]:
        """Extract from and to account IDs."""
        import re
        
        from_match = re.search(r'from\s+(CHK|SAV|CC|checking|savings?|credit)\d*', description, re.IGNORECASE)
        to_match = re.search(r'to\s+(CHK|SAV|CC|checking|savings?|credit)\d*', description, re.IGNORECASE)
        
        from_id = "CHK001"
        to_id = "SAV001"
        
        if from_match:
            from_type = from_match.group(1).lower()
            if "sav" in from_type:
                from_id = "SAV001"
            elif "credit" in from_type or "cc" in from_type:
                from_id = "CC001"
        
        if to_match:
            to_type = to_match.group(1).lower()
            if "sav" in to_type:
                to_id = "SAV001"
            elif "credit" in to_type or "cc" in to_type:
                to_id = "CC001"
            elif "chk" in to_type or "check" in to_type:
                to_id = "CHK001"
        
        return from_id, to_id

    def _extract_biller_name(self, description: str) -> str:
        """Extract biller name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        
        # Common billers
        billers = ["Electric Company", "Water Utility", "Internet Provider", "Phone Bill", "Insurance"]
        desc_lower = description.lower()
        for biller in billers:
            if biller.lower().split()[0] in desc_lower:
                return biller
        
        return random.choice(billers)


class RobinhoodApp(BaseApp):
    """Robinhood investment & trading app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Robinhood", user_id)

    def _initialize_state(self) -> None:
        """Initialize Robinhood state based on RobinhoodState model."""
        self.state = {
            "user_id": self.user_id,
            "cash_balance": round(random.uniform(500, 5000), 2),
            "holdings": [
                {"symbol": "AAPL", "asset_type": "stock", "quantity": random.randint(1, 10), "average_buy_price": round(random.uniform(150, 180), 2)},
                {"symbol": "GOOGL", "asset_type": "stock", "quantity": random.randint(1, 5), "average_buy_price": round(random.uniform(130, 150), 2)},
                {"symbol": "BTC", "asset_type": "crypto", "quantity": round(random.uniform(0.01, 0.5), 4), "average_buy_price": round(random.uniform(40000, 50000), 2)},
            ],
            "watchlist": ["TSLA", "MSFT", "NVDA", "ETH", "AMZN"],
            "transaction_history": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Robinhood API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "GetPortfolio":
            return self._get_portfolio(timestamp, description, context)
        elif api_name == "GetWatchlist":
            return self._get_watchlist(timestamp, description, context)
        elif api_name == "SearchStocks":
            return self._search_stocks(timestamp, description, context)
        elif api_name == "GetStockQuote":
            return self._get_stock_quote(timestamp, description, context)
        elif api_name == "BuyStock":
            return self._buy_stock(timestamp, description, context)
        elif api_name == "SellStock":
            return self._sell_stock(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Robinhood API: {api_name}")

    def _get_portfolio(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View current investment holdings, positions, and portfolio value."""
        # Calculate total portfolio value
        total_value = self.state["cash_balance"]
        for holding in self.state["holdings"]:
            current_price = self._get_current_price(holding["symbol"], holding["asset_type"])
            total_value += holding["quantity"] * current_price

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetPortfolio",
            request={},
            response={
                "cash_balance": round(self.state["cash_balance"], 2),
                "holdings": self.state["holdings"],
                "total_portfolio_value": round(total_value, 2)
            }
        )

    def _get_watchlist(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View list of stocks or crypto being monitored."""
        watchlist_details = []
        for symbol in self.state["watchlist"]:
            asset_type = "crypto" if symbol in ["BTC", "ETH", "DOGE"] else "stock"
            current_price = self._get_current_price(symbol, asset_type)
            change_percent = round(random.uniform(-5, 5), 2)
            watchlist_details.append({
                "symbol": symbol,
                "current_price": current_price,
                "change_percent": change_percent
            })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetWatchlist",
            request={},
            response={
                "watchlist": watchlist_details
            }
        )

    def _search_stocks(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search for stocks or crypto by symbol or company name."""
        query = self._extract_search_query(description)

        # Generate search results
        results = self._generate_search_results(query)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchStocks",
            request={"query": query},
            response={
                "results": results
            }
        )

    def _get_stock_quote(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View current price, change, and details for a specific stock or crypto."""
        symbol = self._extract_symbol(description)
        asset_type = "crypto" if symbol in ["BTC", "ETH", "DOGE", "SOL", "XRP"] else "stock"
        current_price = self._get_current_price(symbol, asset_type)
        change_percent = round(random.uniform(-5, 5), 2)
        in_watchlist = symbol in self.state["watchlist"]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetStockQuote",
            request={"symbol": symbol},
            response={
                "symbol": symbol,
                "current_price": current_price,
                "change_percent": change_percent,
                "timestamp": timestamp,
                "in_watchlist": in_watchlist
            }
        )

    def _buy_stock(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Execute a purchase of stocks or crypto."""
        symbol = self._extract_symbol(description)
        quantity = self._extract_quantity(description)
        asset_type = self._extract_asset_type(description, symbol)
        current_price = self._get_current_price(symbol, asset_type)
        total_cost = quantity * current_price

        # Check if we have enough cash
        if total_cost > self.state["cash_balance"]:
            quantity = self.state["cash_balance"] / current_price
            total_cost = quantity * current_price

        # Deduct cash
        self.state["cash_balance"] -= total_cost

        # Update or create holding
        existing_holding = None
        for holding in self.state["holdings"]:
            if holding["symbol"] == symbol:
                existing_holding = holding
                break

        if existing_holding:
            # Update average price
            old_value = existing_holding["quantity"] * existing_holding["average_buy_price"]
            new_value = quantity * current_price
            new_quantity = existing_holding["quantity"] + quantity
            existing_holding["quantity"] = new_quantity
            existing_holding["average_buy_price"] = round((old_value + new_value) / new_quantity, 2)
            new_holding = existing_holding
        else:
            new_holding = {
                "symbol": symbol,
                "asset_type": asset_type,
                "quantity": quantity,
                "average_buy_price": current_price
            }
            self.state["holdings"].append(new_holding)

        # Record transaction (bounded)
        transaction_id = f"TXN{uuid.uuid4().hex[:8].upper()}"
        transaction = {
            "transaction_id": transaction_id,
            "symbol": symbol,
            "asset_type": asset_type,
            "transaction_type": TransactionType.BUY.value,
            "quantity": quantity,
            "price": current_price,
            "timestamp": timestamp
        }
        self._append_to_history("transaction_history", transaction)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="BuyStock",
            request={"symbol": symbol, "quantity": quantity, "asset_type": asset_type},
            response={
                "transaction": transaction,
                "new_cash_balance": round(self.state["cash_balance"], 2),
                "new_holding": new_holding
            }
        )

    def _sell_stock(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Execute a sale of stocks or crypto."""
        symbol = self._extract_symbol(description)
        quantity = self._extract_quantity(description)
        asset_type = self._extract_asset_type(description, symbol)
        current_price = self._get_current_price(symbol, asset_type)

        # Find holding
        holding = None
        for h in self.state["holdings"]:
            if h["symbol"] == symbol:
                holding = h
                break

        remaining_holding = None
        if holding:
            # Limit quantity to what we have
            quantity = min(quantity, holding["quantity"])
            holding["quantity"] -= quantity
            
            if holding["quantity"] > 0:
                remaining_holding = holding
        else:
            self.state["holdings"].remove(holding)

            # Add cash
            self.state["cash_balance"] += quantity * current_price

        # Record transaction (bounded)
        transaction_id = f"TXN{uuid.uuid4().hex[:8].upper()}"
        transaction = {
            "transaction_id": transaction_id,
            "symbol": symbol,
            "asset_type": asset_type,
            "transaction_type": TransactionType.SELL.value,
            "quantity": quantity,
            "price": current_price,
            "timestamp": timestamp
        }
        self._append_to_history("transaction_history", transaction)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SellStock",
            request={"symbol": symbol, "quantity": quantity, "asset_type": asset_type},
            response={
                "transaction": transaction,
                "new_cash_balance": round(self.state["cash_balance"], 2),
                "remaining_holding": remaining_holding
            }
        )

    def _get_current_price(self, symbol: str, asset_type: str) -> float:
        """Get simulated current price for a symbol."""
        base_prices = {
            "AAPL": 175, "GOOGL": 140, "MSFT": 380, "TSLA": 250, "NVDA": 450,
            "AMZN": 180, "META": 350, "BTC": 45000, "ETH": 2500, "DOGE": 0.08,
            "SOL": 100, "XRP": 0.55
        }
        base = base_prices.get(symbol, 100 if asset_type == "stock" else 50)
        # Add some random variation
        return round(base * random.uniform(0.95, 1.05), 2)

    def _extract_symbol(self, description: str) -> str:
        """Extract stock/crypto symbol from description."""
        import re
        # Look for uppercase symbols
        match = re.search(r'\b([A-Z]{1,5})\b', description)
        if match and match.group(1) not in ["A", "I", "THE", "AND", "FOR"]:
            return match.group(1)
        
        # Check for crypto names
        desc_lower = description.lower()
        if "bitcoin" in desc_lower:
            return "BTC"
        elif "ethereum" in desc_lower:
            return "ETH"
        elif "apple" in desc_lower:
            return "AAPL"
        elif "google" in desc_lower:
            return "GOOGL"
        elif "tesla" in desc_lower:
            return "TSLA"
        
        return random.choice(["AAPL", "GOOGL", "TSLA", "BTC"])

    def _extract_quantity(self, description: str) -> float:
        """Extract quantity from description."""
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:shares?|units?|coins?)?', description, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return random.uniform(1, 10)

    def _extract_asset_type(self, description: str, symbol: str) -> str:
        """Extract or infer asset type."""
        crypto_symbols = ["BTC", "ETH", "DOGE", "SOL", "XRP", "ADA", "DOT"]
        if symbol in crypto_symbols:
            return "crypto"
        if "crypto" in description.lower():
            return "crypto"
        return "stock"

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "stock"

    def _generate_search_results(self, query: str) -> List[Dict]:
        """Generate search results for stocks/crypto."""
        results = []
        # Some common mappings
        stock_map = {
            "apple": ("AAPL", "Apple Inc."),
            "google": ("GOOGL", "Alphabet Inc."),
            "tesla": ("TSLA", "Tesla Inc."),
            "microsoft": ("MSFT", "Microsoft Corp."),
            "amazon": ("AMZN", "Amazon.com Inc."),
            "bitcoin": ("BTC", "Bitcoin"),
            "ethereum": ("ETH", "Ethereum"),
        }
        
        query_lower = query.lower()
        for key, (symbol, name) in stock_map.items():
            if key in query_lower or symbol.lower() in query_lower:
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "current_price": self._get_current_price(symbol, "crypto" if symbol in ["BTC", "ETH"] else "stock")
                })
        
        if not results:
            # Return some default results
            results = [
                {"symbol": "AAPL", "name": "Apple Inc.", "current_price": self._get_current_price("AAPL", "stock")},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "current_price": self._get_current_price("GOOGL", "stock")},
            ]
        
        return results


class GmailApp(BaseApp):
    """Gmail email app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Gmail", user_id)

    def _initialize_state(self) -> None:
        """Initialize Gmail state based on GmailState model."""
        email_address = f"{self.user_id}@gmail.com"
        self.state = {
            "user_id": self.user_id,
            "email_address": email_address,
            "inbox": self._generate_initial_inbox(),
            "sent_emails": []
        }

    def _generate_initial_inbox(self) -> List[Dict]:
        """Generate initial inbox emails."""
        senders = [
            "newsletter@company.com", "support@service.com", "friend@email.com",
            "boss@work.com", "noreply@social.com", "updates@platform.com"
        ]
        subjects = [
            "Weekly Newsletter", "Your recent inquiry", "Hey, catching up!",
            "Project Update Required", "New notification", "Important Update"
        ]
        
        emails = []
        base_date = datetime.now()
        for i in range(6):
            email_id = f"EMAIL{uuid.uuid4().hex[:8].upper()}"
            timestamp = (base_date - timedelta(hours=i*4)).strftime("%Y-%m-%d %H:%M:%S")
            emails.append({
                "email_id": email_id,
                "from_address": senders[i],
                "to_address": self.state.get("email_address", "user@gmail.com") if hasattr(self, 'state') else "user@gmail.com",
                "subject": subjects[i],
                "body": f"This is the content of email about {subjects[i].lower()}...",
                "timestamp": timestamp,
                "is_read": i > 2,
                "labels": ["inbox"]
            })
        return emails

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Gmail API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "GetInbox":
            return self._get_inbox(timestamp, description, context)
        elif api_name == "ReadEmail":
            return self._read_email(timestamp, description, context)
        elif api_name == "SendEmail":
            return self._send_email(timestamp, description, context)
        elif api_name == "ReplyEmail":
            return self._reply_email(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Gmail API: {api_name}")

    def _get_inbox(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Retrieve current inbox emails with previews and metadata."""
        limit = self._extract_limit(description)
        unread_only = "unread" in description.lower()

        if unread_only:
            emails = [e for e in self.state["inbox"] if not e.get("is_read")]
        else:
            emails = self.state["inbox"]

        emails = emails[:limit]
        unread_count = sum(1 for e in self.state["inbox"] if not e.get("is_read"))

        # Create preview format
        email_previews = []
        for email in emails:
            email_previews.append({
                "email_id": email["email_id"],
                "from": email["from_address"],
                "subject": email["subject"],
                "snippet": email["body"][:50] + "..." if len(email["body"]) > 50 else email["body"],
                "timestamp": email["timestamp"],
                "is_read": email.get("is_read", False)
            })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetInbox",
            request={"limit": limit, "unread_only": unread_only},
            response={
                "emails": email_previews,
                "unread_count": unread_count
            }
        )

    def _read_email(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Open and read a specific email."""
        email_id = self._extract_email_id(description)

        # Find email
        email = None
        for e in self.state["inbox"]:
            if e["email_id"] == email_id:
                email = e
                e["is_read"] = True
                break

        if not email:
            # Return most recent if not found
            email = self.state["inbox"][0] if self.state["inbox"] else {
                "email_id": email_id,
                "from_address": "unknown@email.com",
                "to_address": self.state["email_address"],
                "subject": "Email",
                "body": "Email content not found.",
            "timestamp": timestamp,
                "is_read": True,
                "labels": []
        }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ReadEmail",
            request={"email_id": email_id},
            response={
                "email": email
            }
        )

    def _send_email(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Compose and send a new email."""
        to_address = self._extract_recipient(description)
        subject = self._extract_subject(description)
        body = self._extract_body(description)

        email_id = f"EMAIL{uuid.uuid4().hex[:8].upper()}"
        email = {
            "email_id": email_id,
            "from_address": self.state["email_address"],
            "to_address": to_address,
            "subject": subject,
            "body": body,
            "timestamp": timestamp,
            "is_read": True,
            "labels": ["sent"]
        }
        self._append_to_history("sent_emails", email)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SendEmail",
            request={"to": to_address, "subject": subject, "body": body},
            response={
                "email": email,
                "sent_timestamp": timestamp
            }
        )

    def _reply_email(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Reply to a received email."""
        email_id = self._extract_email_id(description)
        body = self._extract_body(description)

        # Find original email
        original_email = None
        for e in self.state["inbox"]:
            if e["email_id"] == email_id:
                original_email = e
                break

        if not original_email and self.state["inbox"]:
            original_email = self.state["inbox"][0]

        reply_id = f"EMAIL{uuid.uuid4().hex[:8].upper()}"
        to_address = original_email["from_address"] if original_email else "unknown@email.com"
        subject = f"Re: {original_email['subject']}" if original_email else "Re: Your email"

        reply_email = {
            "email_id": reply_id,
            "from_address": self.state["email_address"],
            "to_address": to_address,
            "subject": subject,
            "body": body,
            "timestamp": timestamp,
            "is_read": True,
            "labels": ["sent"]
        }
        self._append_to_history("sent_emails", reply_email)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ReplyEmail",
            request={"email_id": email_id, "body": body},
            response={
                "email": reply_email,
                "sent_timestamp": timestamp
            }
        )

    def _extract_limit(self, description: str) -> int:
        """Extract limit from description."""
        import re
        match = re.search(r'(\d+)\s*(?:emails?|messages?)', description, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 50

    def _extract_email_id(self, description: str) -> str:
        """Extract email ID from description."""
        import re
        match = re.search(r'EMAIL[A-Z0-9]+', description)
        if match:
            return match.group(0)
        return f"EMAIL{random.randint(10000, 99999)}"

    def _extract_recipient(self, description: str) -> str:
        """Extract recipient email from description."""
        import re
        # Try to find email pattern
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', description)
        if match:
            return match.group(0)
        
        # Try quoted name
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            name = quoted[0][0] or quoted[0][1]
            return f"{name.lower().replace(' ', '.')}@email.com"
        
        return "recipient@email.com"

    def _extract_subject(self, description: str) -> str:
        """Extract email subject from description."""
        import re
        # Look for "subject:" pattern
        match = re.search(r'subject[:\s]+["\']?([^"\']+)["\']?', description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try quoted text as subject
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        
        return "Message"

    def _extract_body(self, description: str) -> str:
        """Extract email body from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if len(quoted) > 1:
            return quoted[1][0] or quoted[1][1]
        elif quoted:
            return quoted[0][0] or quoted[0][1]
        return description[:200]


class LLMAssistantApp(BaseApp):
    """LLM Assistant app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("LLM Assistant", user_id)

    def _initialize_state(self) -> None:
        """Initialize LLM state based on LLMState model."""
        self.state = {
            "user_id": self.user_id,
            "conversations": []
        }
        self._current_conversation_id: Optional[str] = None

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call LLM Assistant API based on app_catalog.py definitions."""
        self.ensure_initialized()

        if api_name == "CreateConversation":
            return self._create_conversation(timestamp, description, context)
        elif api_name == "ContinueConversation":
            return self._continue_conversation(timestamp, description, context)
        else:
            raise ValueError(f"Unknown LLM Assistant API: {api_name}")

    def _create_conversation(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Start a new conversation thread with the AI assistant."""
        initial_message = self._extract_message(description)
        
        conversation_id = f"CONV{uuid.uuid4().hex[:8].upper()}"
        conversation = {
            "conversation_id": conversation_id,
            "messages": [],
            "created_at": timestamp
        }
        
        self._append_to_history("conversations", conversation)
        self._current_conversation_id = conversation_id

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateConversation",
            request={"initial_message": initial_message},
            response={
                "conversation_id": conversation_id,
                "created_at": timestamp
            }
        )

    def _continue_conversation(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Send a message in an existing conversation thread."""
        conversation_id = self._extract_conversation_id(description)
        user_message = self._extract_message(description)

        # Find conversation or use most recent
        conversation = None
        for conv in self.state["conversations"]:
            if conv["conversation_id"] == conversation_id:
                conversation = conv
                break
        
        if not conversation and self.state["conversations"]:
            conversation = self.state["conversations"][-1]
            conversation_id = conversation["conversation_id"]
        elif not conversation:
            # Create a new conversation if none exists
            conversation_id = f"CONV{uuid.uuid4().hex[:8].upper()}"
            conversation = {
                "conversation_id": conversation_id,
                "messages": [],
                "created_at": timestamp
            }
            self._append_to_history("conversations", conversation)

        # Create user message
        user_msg_id = f"MSG{uuid.uuid4().hex[:8].upper()}"
        user_msg = {
            "message_id": user_msg_id,
            "role": "user",
            "content": user_message,
            "timestamp": timestamp
        }
        conversation["messages"].append(user_msg)

        # Generate AI response
        ai_response = self._generate_ai_response(user_message, conversation, context)
        
        assistant_msg_id = f"MSG{uuid.uuid4().hex[:8].upper()}"
        assistant_msg = {
            "message_id": assistant_msg_id,
            "role": "assistant",
            "content": ai_response,
            "timestamp": timestamp
        }
        conversation["messages"].append(assistant_msg)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ContinueConversation",
            request={"conversation_id": conversation_id, "message": user_message},
            response={
                "conversation_id": conversation_id,
                "user_message": user_msg,
                "assistant_response": assistant_msg
            }
        )

    def _extract_message(self, description: str) -> str:
        """Extract message from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return description

    def _extract_conversation_id(self, description: str) -> str:
        """Extract conversation ID from description."""
        import re
        match = re.search(r'CONV[A-Z0-9]+', description)
        if match:
            return match.group(0)
        return self._current_conversation_id or f"CONV{random.randint(10000, 99999)}"

    def _generate_ai_response(self, user_message: str, conversation: Dict, context: Dict[str, Any]) -> str:
        """Generate a plausible AI response."""
        # Use LLM data generator for chat responses
        data_gen = get_data_generator()
        user_profile = context.get("user_profile", {})

        # Convert conversation messages to history format
        history = [
            {"user_message": msg["content"], "ai_response": ""}
            for msg in conversation.get("messages", [])
            if msg["role"] == "user"
        ]

        response = data_gen.generate_llm_chat_response(
            user_message=user_message,
            conversation_history=history,
            user_profile=user_profile
        )

        return response


class LinkedInApp(BaseApp):
    """LinkedIn professional networking app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("LinkedIn", user_id)

    def _initialize_state(self) -> None:
        """Initialize LinkedIn state based on LinkedInState model."""
        self.state = {
            "user_id": self.user_id,
            "headline": "Professional",
            "summary": "",
            "experiences": [],
            "skills": ["Communication", "Problem Solving", "Teamwork"],
            "connections": [],
            "posts": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call LinkedIn API based on app_catalog.py definitions."""
        self.ensure_initialized()

        api_handlers = {
            "UpdateProfile": self._update_profile,
            "AddExperience": self._add_experience,
            "AddSkill": self._add_skill,
            "PostUpdate": self._post_update,
            "GetFeed": self._get_feed,
            "LikePost": self._like_post,
            "CommentOnPost": self._comment_on_post,
            "SearchJobs": self._search_jobs,
            "ApplyJob": self._apply_job,
            "SendConnectionRequest": self._send_connection_request,
        }
        
        handler = api_handlers.get(api_name)
        if handler:
            return handler(timestamp, description, context)
        raise ValueError(f"Unknown LinkedIn API: {api_name}")

    def _update_profile(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        headline = self._extract_quoted(description) or self.state["headline"]
        summary = description if "summary" in description.lower() else self.state["summary"]
        
        updated_fields = {}
        if headline != self.state["headline"]:
            self.state["headline"] = headline
            updated_fields["headline"] = headline
        if summary != self.state["summary"]:
            self.state["summary"] = summary
            updated_fields["summary"] = summary

        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="UpdateProfile",
            request={"headline": headline, "summary": summary},
            response={"updated_fields": updated_fields})

    def _add_experience(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        company = self._extract_quoted(description) or "Company"
        title = "Position"
        experience = {"company": company, "title": title, "start_date": timestamp.split()[0][:7], "end_date": None}
        self._append_to_history("experiences", experience)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="AddExperience",
            request={"company": company, "title": title, "start_date": experience["start_date"]},
            response={"experience": experience, "total_experiences": len(self.state["experiences"])})

    def _add_skill(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        skill = self._extract_quoted(description) or "New Skill"
        if skill not in self.state["skills"]:
            self._append_to_history("skills", skill)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="AddSkill",
            request={"skill": skill}, response={"skills": self.state["skills"]})

    def _post_update(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        content = self._extract_quoted(description) or description[:200]
        post_id = f"POST{uuid.uuid4().hex[:8].upper()}"
        post = {"post_id": post_id, "author": self.user_id, "content": content, "timestamp": timestamp, "likes_count": 0}
        self._append_to_history("posts", post)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="PostUpdate",
            request={"content": content}, response={"post": post})

    def _get_feed(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        limit = 20
        posts = self.state["posts"][-limit:] if self.state["posts"] else self._generate_feed_posts()
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="GetFeed",
            request={"limit": limit}, response={"posts": posts})

    def _like_post(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        post_id = self._extract_post_id(description)
        new_likes = random.randint(10, 100)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="LikePost",
            request={"post_id": post_id}, response={"post_id": post_id, "new_likes_count": new_likes})

    def _comment_on_post(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        post_id = self._extract_post_id(description)
        comment = self._extract_quoted(description) or "Great post!"
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="CommentOnPost",
            request={"post_id": post_id, "comment": comment},
            response={"post_id": post_id, "comment_timestamp": timestamp})

    def _search_jobs(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        query = self._extract_quoted(description) or "software engineer"
        jobs = [{"job_id": f"JOB{i}", "title": f"{query.title()} Position", "company": f"Company {chr(65+i)}", "location": "Remote"} for i in range(5)]
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="SearchJobs",
            request={"query": query}, response={"jobs": jobs})

    def _apply_job(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        job_id = f"JOB{random.randint(1000, 9999)}"
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="ApplyJob",
            request={"job_id": job_id}, response={"job_id": job_id, "applied_at": timestamp})

    def _send_connection_request(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        user_id = f"USER{random.randint(1000, 9999)}"
        message = self._extract_quoted(description)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="SendConnectionRequest",
            request={"user_id": user_id, "message": message},
            response={"user_id": user_id, "sent_at": timestamp})

    def _extract_quoted(self, description: str) -> Optional[str]:
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        return (quoted[0][0] or quoted[0][1]) if quoted else None

    def _extract_post_id(self, description: str) -> str:
        import re
        match = re.search(r'POST[A-Z0-9]+', description)
        return match.group(0) if match else f"POST{random.randint(1000, 9999)}"

    def _generate_feed_posts(self) -> List[Dict]:
        return [{"post_id": f"POST{i}", "author": f"User{i}", "content": f"Post content {i}", "timestamp": datetime.now().isoformat(), "likes_count": random.randint(0, 100)} for i in range(5)]


class NotionApp(BaseApp):
    """Notion knowledge management & productivity app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Notion", user_id)

    def _initialize_state(self) -> None:
        """Initialize Notion state based on NotionState model."""
        self.state = {
            "user_id": self.user_id,
            "pages": [],
            "database_entries": []
        }

    def call_api(self, api_name: str, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        self.ensure_initialized()
        handlers = {
            "GetPages": self._get_pages,
            "CreatePage": self._create_page,
            "UpdatePage": self._update_page,
            "SearchContent": self._search_content,
            "CreateDatabaseEntry": self._create_database_entry,
        }
        handler = handlers.get(api_name)
        if handler:
            return handler(timestamp, description, context)
        raise ValueError(f"Unknown Notion API: {api_name}")

    def _get_pages(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        pages = [{"page_id": p["page_id"], "title": p["title"], "created_at": p["created_at"], "updated_at": p["updated_at"]} for p in self.state["pages"][:50]]
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="GetPages",
            request={"limit": 50}, response={"pages": pages})

    def _create_page(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        title = (quoted[0][0] or quoted[0][1]) if quoted else "New Page"
        content = description[:500]
        page_id = f"PAGE{uuid.uuid4().hex[:8].upper()}"
        page = {"page_id": page_id, "title": title, "content": content, "created_at": timestamp, "updated_at": timestamp}
        self._append_to_history("pages", page)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="CreatePage",
            request={"title": title, "content": content}, response={"page": page})

    def _update_page(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        import re
        match = re.search(r'PAGE[A-Z0-9]+', description)
        page_id = match.group(0) if match else (self.state["pages"][-1]["page_id"] if self.state["pages"] else f"PAGE{random.randint(1000, 9999)}")
        page = next((p for p in self.state["pages"] if p["page_id"] == page_id), None)
        if page:
            page["updated_at"] = timestamp
            page["content"] = description[:500]
        else:
            page = {"page_id": page_id, "title": "Updated Page", "content": description[:500], "created_at": timestamp, "updated_at": timestamp}
            self._append_to_history("pages", page)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="UpdatePage",
            request={"page_id": page_id}, response={"page": page})

    def _search_content(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        query = (quoted[0][0] or quoted[0][1]) if quoted else "search"
        results = [{"page_id": p["page_id"], "title": p["title"], "snippet": p["content"][:50]} for p in self.state["pages"] if query.lower() in p["title"].lower() or query.lower() in p["content"].lower()]
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="SearchContent",
            request={"query": query}, response={"results": results})

    def _create_database_entry(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        database_name = "tasks" if "task" in description.lower() else "projects" if "project" in description.lower() else "habits"
        entry_id = f"ENTRY{uuid.uuid4().hex[:8].upper()}"
        properties = {"name": description[:50], "status": "active"}
        entry = {"entry_id": entry_id, "database_name": database_name, "properties": properties, "created_at": timestamp}
        self._append_to_history("database_entries", entry)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="CreateDatabaseEntry",
            request={"database_name": database_name, "properties": properties}, response={"entry": entry})


class NetflixApp(BaseApp):
    """Netflix video streaming app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Netflix", user_id)

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "subscription_plan": random.choice(["Basic", "Standard", "Premium"]),
            "my_list": [],
            "watch_history": []
        }
        self._title_cache: Dict[str, Dict] = {}

    def call_api(self, api_name: str, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        self.ensure_initialized()
        handlers = {
            "SearchContent": self._search_content,
            "ShowTitle": self._show_title,
            "PlayContent": self._play_content,
            "AddToMyList": self._add_to_my_list,
            "RateContent": self._rate_content,
        }
        handler = handlers.get(api_name)
        if handler:
            return handler(timestamp, description, context)
        raise ValueError(f"Unknown Netflix API: {api_name}")

    def _search_content(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        query = (quoted[0][0] or quoted[0][1]) if quoted else "movie"
        titles = self._generate_titles(query)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="SearchContent",
            request={"query": query}, response={"titles": titles})

    def _show_title(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        title_id = self._extract_title_id(description)
        title = self._get_or_create_title(title_id, description)
        in_my_list = title_id in self.state["my_list"]
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="ShowTitle",
            request={"title_id": title_id},
            response={"title": title, "description": f"A {title['genre']} {title['content_type']}", "rating": round(random.uniform(3.5, 5.0), 1), "in_my_list": in_my_list})

    def _play_content(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        title_id = self._extract_title_id(description)
        title = self._get_or_create_title(title_id, description)
        watch_record = {"title_id": title_id, "watched_at": timestamp, "duration_watched": random.randint(30, 120), "completed": random.choice([True, False])}
        self._append_to_history("watch_history", watch_record)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="PlayContent",
            request={"title_id": title_id},
            response={"title": title, "playing_status": PlayingStatus.PLAYING.value, "subscription_plan": self.state["subscription_plan"], "play_started_at": timestamp})

    def _add_to_my_list(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        title_id = self._extract_title_id(description)
        if title_id not in self.state["my_list"]:
            self._append_to_history("my_list", title_id)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="AddToMyList",
            request={"title_id": title_id}, response={"my_list": self.state["my_list"]})

    def _rate_content(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        title_id = self._extract_title_id(description)
        rating = 1 if "up" in description.lower() or "good" in description.lower() else 0
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="RateContent",
            request={"title_id": title_id, "rating": rating}, response={"title_id": title_id, "rating": rating})

    def _extract_title_id(self, description: str) -> str:
        import re
        match = re.search(r'TITLE[A-Z0-9]+', description)
        return match.group(0) if match else f"TITLE{random.randint(1000, 9999)}"

    def _generate_titles(self, query: str) -> List[Dict]:
        genres = ["Drama", "Comedy", "Action", "Thriller", "Documentary", "Sci-Fi"]
        titles = []
        for i in range(5):
            title_id = f"TITLE{abs(hash(query + str(i))) % 10000:04d}"
            title = {"title_id": title_id, "name": f"{query.title()} {i+1}", "content_type": random.choice(["movie", "series"]), "genre": random.choice(genres)}
            titles.append(title)
            self._title_cache[title_id] = title
        return titles

    def _get_or_create_title(self, title_id: str, description: str) -> Dict:
        if title_id in self._title_cache:
            return self._title_cache[title_id]
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        name = (quoted[0][0] or quoted[0][1]) if quoted else f"Title {title_id}"
        title = {"title_id": title_id, "name": name, "content_type": "movie", "genre": "Drama"}
        self._title_cache[title_id] = title
        return title


class GoodreadsApp(BaseApp):
    """Goodreads book tracking & reviews app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Goodreads", user_id)

    def _initialize_state(self) -> None:
        self.state = {"user_id": self.user_id, "shelves": [], "reviews": []}
        self._book_cache: Dict[str, Dict] = {}

    def call_api(self, api_name: str, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        self.ensure_initialized()
        handlers = {
            "SearchBooks": self._search_books,
            "ShowBook": self._show_book,
            "AddToShelf": self._add_to_shelf,
            "RateBook": self._rate_book,
            "WriteReview": self._write_review,
        }
        handler = handlers.get(api_name)
        if handler:
            return handler(timestamp, description, context)
        raise ValueError(f"Unknown Goodreads API: {api_name}")

    def _search_books(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        query = (quoted[0][0] or quoted[0][1]) if quoted else "book"
        books = self._generate_books(query)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="SearchBooks",
            request={"query": query}, response={"books": books})

    def _show_book(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        book_id = self._extract_book_id(description)
        book = self._get_or_create_book(book_id, description)
        on_shelf = next((s["shelf"] for s in self.state["shelves"] if s["book_id"] == book_id), None)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="ShowBook",
            request={"book_id": book_id},
            response={"book": book, "description": f"A {book['genre']} book", "average_rating": round(random.uniform(3.5, 4.5), 2), "on_shelf": on_shelf})

    def _add_to_shelf(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        book_id = self._extract_book_id(description)
        shelf = "want-to-read" if "want" in description.lower() else "currently-reading" if "current" in description.lower() else "read"
        shelf_entry = {"book_id": book_id, "shelf": shelf, "added_at": timestamp}
        self.state["shelves"] = [s for s in self.state["shelves"] if s["book_id"] != book_id]
        self._append_to_history("shelves", shelf_entry)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="AddToShelf",
            request={"book_id": book_id, "shelf": shelf}, response={"shelf_entry": shelf_entry})

    def _rate_book(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        book_id = self._extract_book_id(description)
        import re
        match = re.search(r'(\d)', description)
        rating = int(match.group(1)) if match else random.randint(3, 5)
        rating = max(1, min(5, rating))
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="RateBook",
            request={"book_id": book_id, "rating": rating}, response={"rating": rating, "rated_at": timestamp})

    def _write_review(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        book_id = self._extract_book_id(description)
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        review_text = (quoted[0][0] or quoted[0][1]) if quoted else "Great book!"
        review = {"book_id": book_id, "rating": random.randint(3, 5), "review_text": review_text, "reviewed_at": timestamp}
        self._append_to_history("reviews", review)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="WriteReview",
            request={"book_id": book_id, "review_text": review_text}, response={"review": review})

    def _extract_book_id(self, description: str) -> str:
        import re
        match = re.search(r'BOOK[A-Z0-9]+', description)
        return match.group(0) if match else f"BOOK{random.randint(1000, 9999)}"

    def _generate_books(self, query: str) -> List[Dict]:
        genres = ["Fiction", "Non-Fiction", "Mystery", "Science Fiction", "Biography", "Self-Help"]
        books = []
        for i in range(5):
            book_id = f"BOOK{abs(hash(query + str(i))) % 10000:04d}"
            book = {"book_id": book_id, "title": f"{query.title()} Book {i+1}", "author": f"Author {chr(65+i)}", "genre": random.choice(genres)}
            books.append(book)
            self._book_cache[book_id] = book
        return books

    def _get_or_create_book(self, book_id: str, description: str) -> Dict:
        if book_id in self._book_cache:
            return self._book_cache[book_id]
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        title = (quoted[0][0] or quoted[0][1]) if quoted else f"Book {book_id}"
        book = {"book_id": book_id, "title": title, "author": "Unknown Author", "genre": "Fiction"}
        self._book_cache[book_id] = book
        return book


class InstagramApp(BaseApp):
    """Instagram social media & photo sharing app based on app_catalog.py definitions."""

    def __init__(self, user_id: str):
        super().__init__("Instagram", user_id)

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "followers": [f"user{i}" for i in range(random.randint(50, 500))],
            "following": [f"user{i}" for i in range(random.randint(100, 300))],
            "posts": []
        }

    def call_api(self, api_name: str, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        self.ensure_initialized()
        handlers = {
            "PostStory": self._post_story,
            "LikePost": self._like_post,
            "CommentOnPost": self._comment_on_post,
            "SendDirectMessage": self._send_direct_message,
            "FollowUser": self._follow_user,
            "UnfollowUser": self._unfollow_user,
            "GetFollowing": self._get_following,
        }
        handler = handlers.get(api_name)
        if handler:
            return handler(timestamp, description, context)
        raise ValueError(f"Unknown Instagram API: {api_name}")

    def _post_story(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        content_type = "video" if "video" in description.lower() else "photo"
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        caption = (quoted[0][0] or quoted[0][1]) if quoted else None
        post_id = f"POST{uuid.uuid4().hex[:8].upper()}"
        post = {"post_id": post_id, "author": self.user_id, "content_type": "story", "caption": caption or "", "timestamp": timestamp, "likes_count": 0}
        self._append_to_history("posts", post)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="PostStory",
            request={"content_type": content_type, "caption": caption}, response={"post": post})

    def _like_post(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        import re
        match = re.search(r'POST[A-Z0-9]+', description)
        post_id = match.group(0) if match else f"POST{random.randint(1000, 9999)}"
        new_likes = random.randint(10, 1000)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="LikePost",
            request={"post_id": post_id}, response={"post_id": post_id, "new_likes_count": new_likes})

    def _comment_on_post(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        import re
        match = re.search(r'POST[A-Z0-9]+', description)
        post_id = match.group(0) if match else f"POST{random.randint(1000, 9999)}"
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        comment = (quoted[0][0] or quoted[0][1]) if quoted else "Nice!"
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="CommentOnPost",
            request={"post_id": post_id, "comment": comment},
            response={"post_id": post_id, "comment_timestamp": timestamp})

    def _send_direct_message(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        message = (quoted[0][0] or quoted[0][1]) if quoted else "Hey!"
        to_user_id = f"user{random.randint(1, 1000)}"
        message_id = f"DM{uuid.uuid4().hex[:8].upper()}"
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="SendDirectMessage",
            request={"to_user_id": to_user_id, "message": message},
            response={"message_id": message_id, "sent_timestamp": timestamp})

    def _follow_user(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        user_id = f"user{random.randint(1, 10000)}"
        if user_id not in self.state["following"]:
            self._append_to_history("following", user_id)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="FollowUser",
            request={"user_id": user_id}, response={"following": self.state["following"][-10:]})

    def _unfollow_user(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        user_id = self.state["following"][-1] if self.state["following"] else f"user{random.randint(1, 1000)}"
        if user_id in self.state["following"]:
            self.state["following"].remove(user_id)
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="UnfollowUser",
            request={"user_id": user_id}, response={"success": True, "following": self.state["following"][-10:]})

    def _get_following(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        return AppLogEntry(timestamp=timestamp, app_name=self.app_name, api_name="GetFollowing",
            request={}, response={"following": self.state["following"][-50:], "following_count": len(self.state["following"])})


class AppRegistry:
    """
    Registry for all app instances.

    Maintains one instance per (app_name, user_id) to ensure state consistency
    across all domains and time windows.

    Apps based on app_catalog.py definitions:
    - Amazon: E-commerce
    - Spotify: Music Streaming
    - Fitbit: Health & Fitness Tracking
    - Chase: Banking & Financial Management
    - Robinhood: Investment & Trading
    - WhatsApp: Instant Messaging
    - Gmail: Email
    - LinkedIn: Professional Networking
    - Notion: Knowledge Management & Productivity
    - Netflix: Video Streaming
    - Goodreads: Book Tracking & Reviews
    - Instagram: Social Media & Photo Sharing
    - Google: Search Engine
    - LLM Assistant: AI Assistant
    """

    # Mapping of app names to their corresponding classes
    APP_CLASSES: Dict[str, type] = {
        "Amazon": AmazonApp,
        "Spotify": SpotifyApp,
        "Fitbit": FitbitApp,
        "Chase": ChaseApp,
        "Robinhood": RobinhoodApp,
        "WhatsApp": WhatsAppApp,
        "Gmail": GmailApp,
        "LinkedIn": LinkedInApp,
        "Notion": NotionApp,
        "Netflix": NetflixApp,
        "Goodreads": GoodreadsApp,
        "Instagram": InstagramApp,
        "Google": GoogleApp,
        "LLM Assistant": LLMAssistantApp,
    }

    def __init__(self):
        self.apps: Dict[Tuple[str, str], BaseApp] = {}

    def get_app(self, app_name: str, user_id: str) -> BaseApp:
        """Get or create an app instance for this user."""
        key = (app_name, user_id)

        if key not in self.apps:
            app_class = self.APP_CLASSES.get(app_name)
            if app_class:
                self.apps[key] = app_class(user_id)
            else:
                raise ValueError(f"Unknown app: {app_name}. Available apps: {list(self.APP_CLASSES.keys())}")

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

    @classmethod
    def get_available_apps(cls) -> List[str]:
        """Return list of all available app names."""
        return list(cls.APP_CLASSES.keys())
