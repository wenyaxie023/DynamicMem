"""
App system for LLM-driven app log generation.

Each app class owns:
- its API schema bindings (Pydantic input/output models)
- its state initialization
- its state update rules based on LLM-generated request/response
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import app_models


MAX_API_CALL_HISTORY = 20
MAX_HISTORY_SMALL = 10
MAX_HISTORY_MEDIUM = 15
MAX_HISTORY_LARGE = 20


def _model_to_schema(model_class) -> Dict[str, Any]:
    if model_class is None:
        return {}
    if hasattr(model_class, "model_json_schema"):
        return model_class.model_json_schema()
    if hasattr(model_class, "schema"):
        return model_class.schema()
    return {}


class BaseApp(ABC):
    APP_NAME = ""
    API_MODELS: Dict[str, Tuple[type | None, type | None]] = {}

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.state: Dict[str, Any] = {}
        self._initialized = False

    def ensure_initialized(self) -> None:
        if not self._initialized:
            self._initialize_state()
            self._initialized = True

    @abstractmethod
    def _initialize_state(self) -> None:
        raise NotImplementedError

    def record_api_call(
        self,
        event: Dict[str, Any],
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> None:
        self.ensure_initialized()
        self._record_common_state(event, request_payload, response_payload)
        self._update_state(event, request_payload, response_payload)

    def _record_common_state(
        self,
        event: Dict[str, Any],
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> None:
        history = self.state.setdefault("api_call_history", [])
        history.append(
            {
                "timestamp": event.get("timestamp"),
                "app_name": event.get("app_name"),
                "api_name": event.get("api_name"),
                "request": request_payload,
                "response": response_payload,
            }
        )
        if len(history) > MAX_API_CALL_HISTORY:
            self.state["api_call_history"] = history[-MAX_API_CALL_HISTORY:]
        self.state["last_api_call"] = self.state["api_call_history"][-1]

        for key in ("session_id", "conversation_id", "device_id", "account_id", "order_id"):
            if key in response_payload and not self.state.get(key):
                self.state[key] = response_payload[key]
            if key in request_payload and not self.state.get(key):
                self.state[key] = request_payload[key]

    def _append_to_state_history(self, key: str, item: Any, max_length: int) -> None:
        history = self.state.setdefault(key, [])
        history.append(item)
        if len(history) > max_length:
            self.state[key] = history[-max_length:]

    def _trim_state_history(self, key: str, max_length: int) -> None:
        if key in self.state and isinstance(self.state[key], list) and len(self.state[key]) > max_length:
            self.state[key] = self.state[key][-max_length:]

    @abstractmethod
    def _update_state(
        self,
        event: Dict[str, Any],
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> None:
        raise NotImplementedError


class AmazonApp(BaseApp):
    APP_NAME = "Amazon"
    API_MODELS = {
        "SearchProducts": (app_models.SearchProductsInput, app_models.SearchProductsOutput),
        "ShowProduct": (app_models.ShowProductInput, app_models.ShowProductOutput),
        "AddToCart": (app_models.AddToCartInput, app_models.AddToCartOutput),
        "ShowCart": (app_models.ShowCartInput, app_models.ShowCartOutput),
        "ShowWishlist": (app_models.ShowWishlistInput, app_models.ShowWishlistOutput),
        "Checkout": (app_models.CheckoutInput, app_models.CheckoutOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "search_history": [],
            "viewed_products": [],
            "cart": [],
            "wishlist": [],
            "order_history": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name == "SearchProducts":
            query = request_payload.get("query", "")
            self._append_to_state_history("search_history", query, MAX_HISTORY_LARGE)
        elif api_name == "ShowProduct":
            product = response_payload.get("product", {})
            if product:
                viewed_ids = [p.get("product_id") for p in self.state.get("viewed_products", [])]
                if product.get("product_id") not in viewed_ids:
                    self._append_to_state_history("viewed_products", product, MAX_HISTORY_MEDIUM)
        elif api_name == "AddToCart":
            product_id = request_payload.get("product_id", "")
            quantity = request_payload.get("quantity", 1)
            cart = self.state.setdefault("cart", [])
            found = False
            for item in cart:
                if item.get("product_id") == product_id:
                    item["quantity"] = item.get("quantity", 1) + quantity
                    found = True
                    break
            if not found:
                cart.append({"product_id": product_id, "quantity": quantity})
            self._trim_state_history("cart", MAX_HISTORY_SMALL)
        elif api_name == "AddToWishlist":
            product_id = request_payload.get("product_id", "")
            wishlist = self.state.setdefault("wishlist", [])
            if product_id not in [p.get("product_id") for p in wishlist]:
                self._append_to_state_history("wishlist", {"product_id": product_id}, MAX_HISTORY_MEDIUM)
        elif api_name == "Checkout":
            order = response_payload.get("order", {})
            if order:
                self._append_to_state_history("order_history", order, MAX_HISTORY_SMALL)
            self.state["cart"] = []


class SpotifyApp(BaseApp):
    APP_NAME = "Spotify"
    API_MODELS = {
        "SearchSongs": (app_models.SearchSongsInput, app_models.SearchSongsOutput),
        "PlaySong": (app_models.PlaySongInput, app_models.PlaySongOutput),
        "AddToPlaylist": (app_models.AddToPlaylistInput, app_models.AddToPlaylistOutput),
        "FollowArtist": (app_models.FollowArtistInput, app_models.FollowArtistOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "play_history": [],
            "followed_artists": [],
            "playlists": [],
            "favorite_genres": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        timestamp = event.get("timestamp", "")
        if api_name == "PlaySong":
            song = response_payload.get("song", {})
            if song:
                play_record = {
                    "song_id": song.get("song_id", ""),
                    "played_at": timestamp,
                    "duration_played_minutes": song.get("duration_minutes", 3),
                }
                self._append_to_state_history("play_history", play_record, MAX_HISTORY_MEDIUM)
        elif api_name == "FollowArtist":
            artist_id = request_payload.get("artist_id", "")
            if artist_id:
                followed = self.state.setdefault("followed_artists", [])
                if artist_id not in followed:
                    self._append_to_state_history("followed_artists", artist_id, MAX_HISTORY_LARGE)
        elif api_name == "AddToPlaylist":
            self._trim_state_history("playlists", MAX_HISTORY_SMALL)


class FitbitApp(BaseApp):
    APP_NAME = "Fitbit"
    API_MODELS = {
        "LogWorkout": (app_models.LogWorkoutInput, app_models.LogWorkoutOutput),
        "RecordActivity": (app_models.RecordActivityInput, app_models.RecordActivityOutput),
        "SyncDevice": (app_models.SyncDeviceInput, app_models.SyncDeviceOutput),
        "SetGoals": (app_models.SetGoalsInput, app_models.SetGoalsOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "workout_history": [],
            "activity_history": [],
            "daily_syncs": [],
            "goals": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name == "LogWorkout":
            workout = response_payload.get("workout", {})
            if workout:
                self._append_to_state_history("workout_history", workout, MAX_HISTORY_MEDIUM)
        elif api_name == "RecordActivity":
            activity = response_payload.get("activity", {})
            if activity:
                self._append_to_state_history("activity_history", activity, MAX_HISTORY_MEDIUM)
        elif api_name == "SyncDevice":
            sync_data = response_payload.get("sync_data", {})
            if sync_data:
                self._append_to_state_history("daily_syncs", sync_data, MAX_HISTORY_SMALL)
        elif api_name == "SetGoals":
            self._trim_state_history("goals", MAX_HISTORY_SMALL)


class ChaseApp(BaseApp):
    APP_NAME = "Chase"
    API_MODELS = {
        "GetBalance": (app_models.GetBalanceInput, app_models.GetBalanceOutput),
        "GetTransactions": (app_models.GetTransactionsInput, app_models.GetTransactionsOutput),
        "SearchTransactions": (app_models.SearchTransactionsInput, app_models.SearchTransactionsOutput),
        "TransferMoney": (app_models.TransferMoneyInput, app_models.TransferMoneyOutput),
        "PayBill": (app_models.PayBillInput, app_models.PayBillOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "transaction_history": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        timestamp = event.get("timestamp", "")
        if api_name == "PayBill":
            transaction = {
                "transaction_id": response_payload.get("transaction_id", ""),
                "transaction_date": timestamp.split()[0] if timestamp else "",
                "merchant": request_payload.get("biller_name", ""),
                "amount": request_payload.get("amount", 0),
                "transaction_type": "debit",
                "category": "bills",
            }
            self._append_to_state_history("transaction_history", transaction, MAX_HISTORY_SMALL)
        elif api_name == "TransferMoney":
            transaction = {
                "transaction_id": response_payload.get("transaction_id", ""),
                "transaction_date": timestamp.split()[0] if timestamp else "",
                "amount": request_payload.get("amount", 0),
                "transaction_type": "transfer",
                "from_account": request_payload.get("from_account_id", ""),
                "to_account": request_payload.get("to_account_id", ""),
            }
            self._append_to_state_history("transaction_history", transaction, MAX_HISTORY_SMALL)


class RobinhoodApp(BaseApp):
    APP_NAME = "Robinhood"
    API_MODELS = {
        "GetPortfolio": (app_models.GetPortfolioInput, app_models.GetPortfolioOutput),
        "GetWatchlist": (app_models.GetWatchlistInput, app_models.GetWatchlistOutput),
        "SearchStocks": (app_models.SearchStocksInput, app_models.SearchStocksOutput),
        "GetStockQuote": (app_models.GetStockQuoteInput, app_models.GetStockQuoteOutput),
        "BuyStock": (app_models.BuyStockInput, app_models.BuyStockOutput),
        "SellStock": (app_models.SellStockInput, app_models.SellStockOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "transaction_history": [],
            "watchlist": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name in ("BuyStock", "SellStock"):
            transaction = response_payload.get("transaction", {})
            if transaction:
                self._append_to_state_history("transaction_history", transaction, MAX_HISTORY_SMALL)
        self._trim_state_history("watchlist", MAX_HISTORY_LARGE)


class WhatsAppApp(BaseApp):
    APP_NAME = "WhatsApp"
    API_MODELS = {
        "GetMessages": (app_models.GetMessagesInput, app_models.GetMessagesOutput),
        "SendMessage": (app_models.SendMessageInput, app_models.SendMessageOutput),
        "SendMedia": (app_models.SendMediaInput, app_models.SendMediaOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "message_history": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name in ("SendMessage", "SendMedia"):
            message = response_payload.get("message", {})
            if message:
                self._append_to_state_history("message_history", message, MAX_HISTORY_MEDIUM)
        elif api_name == "GetMessages":
            self._trim_state_history("message_history", MAX_HISTORY_MEDIUM)


class GmailApp(BaseApp):
    APP_NAME = "Gmail"
    API_MODELS = {
        "GetInbox": (app_models.GetInboxInput, app_models.GetInboxOutput),
        "ReadEmail": (app_models.ReadEmailInput, app_models.ReadEmailOutput),
        "SendEmail": (app_models.SendEmailInput, app_models.SendEmailOutput),
        "ReplyEmail": (app_models.ReplyEmailInput, app_models.ReplyEmailOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "inbox": [],
            "sent_emails": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name in ("SendEmail", "ReplyEmail"):
            email = response_payload.get("email", {})
            if email:
                self._append_to_state_history("sent_emails", email, MAX_HISTORY_SMALL)
        self._trim_state_history("inbox", MAX_HISTORY_SMALL)


class LinkedInApp(BaseApp):
    APP_NAME = "LinkedIn"
    API_MODELS = {
        "UpdateProfile": (app_models.UpdateProfileInput, app_models.UpdateProfileOutput),
        "AddExperience": (app_models.AddExperienceInput, app_models.AddExperienceOutput),
        "AddSkill": (app_models.AddSkillInput, app_models.AddSkillOutput),
        "PostUpdate": (app_models.PostUpdateInput, app_models.PostUpdateOutput),
        "GetFeed": (app_models.GetFeedInput, app_models.GetFeedOutput),
        "LikePost": (app_models.LikePostInput, app_models.LikePostOutput),
        "CommentOnPost": (app_models.CommentOnPostInput, app_models.CommentOnPostOutput),
        "SearchJobs": (app_models.SearchJobsInput, app_models.SearchJobsOutput),
        "ApplyJob": (app_models.ApplyJobInput, app_models.ApplyJobOutput),
        "SendConnectionRequest": (app_models.SendConnectionRequestInput, app_models.SendConnectionRequestOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "posts": [],
            "experiences": [],
            "skills": [],
            "connections": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name == "PostUpdate":
            post = response_payload.get("post", {})
            if post:
                self._append_to_state_history("posts", post, MAX_HISTORY_MEDIUM)
        elif api_name == "AddExperience":
            experience = response_payload.get("experience", {})
            if experience:
                self._append_to_state_history("experiences", experience, MAX_HISTORY_SMALL)
        elif api_name == "AddSkill":
            skill = request_payload.get("skill", "")
            if skill:
                skills = self.state.setdefault("skills", [])
                if skill not in skills:
                    self._append_to_state_history("skills", skill, MAX_HISTORY_LARGE)
        elif api_name == "SendConnectionRequest":
            user_id = request_payload.get("user_id", "")
            if user_id:
                self._append_to_state_history("connections", user_id, MAX_HISTORY_LARGE)


class NotionApp(BaseApp):
    APP_NAME = "Notion"
    API_MODELS = {
        "GetPages": (app_models.GetPagesInput, app_models.GetPagesOutput),
        "CreatePage": (app_models.CreatePageInput, app_models.CreatePageOutput),
        "UpdatePage": (app_models.UpdatePageInput, app_models.UpdatePageOutput),
        "SearchContent": (app_models.NotionSearchContentInput, app_models.NotionSearchContentOutput),
        "CreateDatabaseEntry": (app_models.CreateDatabaseEntryInput, app_models.CreateDatabaseEntryOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "pages": [],
            "database_entries": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name == "CreatePage":
            page = response_payload.get("page", {})
            if page:
                self._append_to_state_history("pages", page, MAX_HISTORY_SMALL)
        elif api_name == "UpdatePage":
            page = response_payload.get("page", {})
            if page:
                pages = self.state.get("pages", [])
                updated = False
                for idx, existing in enumerate(pages):
                    if existing.get("page_id") == page.get("page_id"):
                        pages[idx] = page
                        updated = True
                        break
                if not updated:
                    self._append_to_state_history("pages", page, MAX_HISTORY_SMALL)
        elif api_name == "CreateDatabaseEntry":
            entry = response_payload.get("entry", {})
            if entry:
                self._append_to_state_history("database_entries", entry, MAX_HISTORY_SMALL)


class NetflixApp(BaseApp):
    APP_NAME = "Netflix"
    API_MODELS = {
        "ShowTitle": (app_models.ShowTitleInput, app_models.ShowTitleOutput),
        "PlayContent": (app_models.PlayContentInput, app_models.PlayContentOutput),
        "AddToMyList": (app_models.AddToMyListInput, app_models.AddToMyListOutput),
        "RateContent": (app_models.RateContentInput, app_models.RateContentOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "watch_history": [],
            "my_list": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        timestamp = event.get("timestamp", "")
        if api_name == "PlayContent":
            watch_record = {
                "title_id": request_payload.get("title_id", ""),
                "watched_at": timestamp,
            }
            self._append_to_state_history("watch_history", watch_record, MAX_HISTORY_MEDIUM)
        elif api_name == "AddToMyList":
            title_id = request_payload.get("title_id", "")
            my_list = self.state.setdefault("my_list", [])
            if title_id and title_id not in my_list:
                self._append_to_state_history("my_list", title_id, MAX_HISTORY_LARGE)


class GoodreadsApp(BaseApp):
    APP_NAME = "Goodreads"
    API_MODELS = {
        "SearchBooks": (app_models.SearchBooksInput, app_models.SearchBooksOutput),
        "ShowBook": (app_models.ShowBookInput, app_models.ShowBookOutput),
        "AddToShelf": (app_models.AddToShelfInput, app_models.AddToShelfOutput),
        "RateBook": (app_models.RateBookInput, app_models.RateBookOutput),
        "WriteReview": (app_models.WriteReviewInput, app_models.WriteReviewOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "shelves": [],
            "reviews": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name == "AddToShelf":
            shelf_entry = response_payload.get("shelf_entry", {})
            if shelf_entry:
                self._append_to_state_history("shelves", shelf_entry, MAX_HISTORY_MEDIUM)
        elif api_name == "WriteReview":
            review = response_payload.get("review", {})
            if review:
                self._append_to_state_history("reviews", review, MAX_HISTORY_SMALL)


class InstagramApp(BaseApp):
    APP_NAME = "Instagram"
    API_MODELS = {
        "CreatePost": (app_models.CreatePostInput, app_models.CreatePostOutput),
        "PostStory": (app_models.PostStoryInput, app_models.PostStoryOutput),
        "SendDirectMessage": (app_models.SendDirectMessageInput, app_models.SendDirectMessageOutput),
        "FollowUser": (app_models.FollowUserInput, app_models.FollowUserOutput),
        "UnfollowUser": (app_models.UnfollowUserInput, app_models.UnfollowUserOutput),
        "GetFollowing": (app_models.GetFollowingInput, app_models.GetFollowingOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "posts": [],
            "following": [],
            "followers": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        if api_name in ("CreatePost", "PostStory"):
            post = response_payload.get("post", {})
            if post:
                self._append_to_state_history("posts", post, MAX_HISTORY_MEDIUM)
        elif api_name == "FollowUser":
            user_id = request_payload.get("user_id", "")
            following = self.state.setdefault("following", [])
            if user_id and user_id not in following:
                self._append_to_state_history("following", user_id, MAX_HISTORY_LARGE)
        elif api_name == "UnfollowUser":
            user_id = request_payload.get("user_id", "")
            following = self.state.get("following", [])
            if user_id in following:
                following.remove(user_id)
        self._trim_state_history("followers", MAX_HISTORY_LARGE)


class GoogleApp(BaseApp):
    APP_NAME = "Google"
    API_MODELS = {
        "Search": (app_models.GoogleSearchInput, app_models.GoogleSearchOutput),
        "ClickResult": (app_models.ClickResultInput, app_models.ClickResultOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "search_history": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        timestamp = event.get("timestamp", "")
        if api_name == "Search":
            query = request_payload.get("query", "")
            results = response_payload.get("results", [])
            search_record = {
                "query": query,
                "results": results[:3],
                "searched_at": timestamp,
                "clicked_result_id": None,
            }
            self._append_to_state_history("search_history", search_record, MAX_HISTORY_SMALL)
        elif api_name == "ClickResult":
            result_id = request_payload.get("result_id", "")
            search_query = request_payload.get("search_query", "")
            for record in reversed(self.state.get("search_history", [])):
                if record.get("query") == search_query:
                    record["clicked_result_id"] = result_id
                    break


class LLMAssistantApp(BaseApp):
    APP_NAME = "LLM Assistant"
    API_MODELS = {
        "CreateConversation": (app_models.CreateConversationInput, app_models.CreateConversationOutput),
        "ContinueConversation": (app_models.ContinueConversationInput, app_models.ContinueConversationOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "conversations": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        timestamp = event.get("timestamp", "")
        if api_name == "CreateConversation":
            conv_id = response_payload.get("conversation_id", "")
            if conv_id:
                conversation = {
                    "conversation_id": conv_id,
                    "messages": [],
                    "created_at": timestamp,
                }
                self._append_to_state_history("conversations", conversation, MAX_HISTORY_SMALL)
        elif api_name == "ContinueConversation":
            conv_id = request_payload.get("conversation_id", "")
            user_msg = response_payload.get("user_message", {})
            assistant_msg = response_payload.get("assistant_response", {})
            for conv in self.state.get("conversations", []):
                if conv.get("conversation_id") == conv_id:
                    messages = conv.setdefault("messages", [])
                    if user_msg:
                        messages.append(user_msg)
                    if assistant_msg:
                        messages.append(assistant_msg)
                    if len(messages) > MAX_HISTORY_SMALL * 2:
                        conv["messages"] = messages[-(MAX_HISTORY_SMALL * 2):]
                    break


class GoogleMapsApp(BaseApp):
    APP_NAME = "Google Maps"
    API_MODELS = {
        "ShareLocation": (app_models.ShareLocationInput, app_models.ShareLocationOutput),
        "CheckIn": (app_models.CheckInInput, app_models.CheckInOutput),
        "GetDirections": (app_models.GetDirectionsInput, app_models.GetDirectionsOutput),
        "SearchPlaces": (app_models.SearchPlacesInput, app_models.SearchPlacesOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "shared_locations": [],
            "checkin_history": [],
            "directions_history": [],
            "place_search_history": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        timestamp = event.get("timestamp", "")
        if api_name == "ShareLocation":
            shared_location = response_payload.get("shared_location", {})
            if shared_location:
                self._append_to_state_history("shared_locations", shared_location, MAX_HISTORY_MEDIUM)
        elif api_name == "CheckIn":
            checkin = response_payload.get("checkin", {})
            if checkin:
                self._append_to_state_history("checkin_history", checkin, MAX_HISTORY_MEDIUM)
        elif api_name == "GetDirections":
            directions = response_payload.get("directions", {})
            if directions:
                self._append_to_state_history("directions_history", directions, MAX_HISTORY_MEDIUM)
        elif api_name == "SearchPlaces":
            query = request_payload.get("query", "")
            if query:
                search_record = {
                    "query": query,
                    "category": request_payload.get("category"),
                    "searched_at": timestamp,
                }
                self._append_to_state_history("place_search_history", search_record, MAX_HISTORY_SMALL)


class UberEatsApp(BaseApp):
    APP_NAME = "UberEats"
    API_MODELS = {
        "SearchRestaurants": (app_models.SearchRestaurantsInput, app_models.SearchRestaurantsOutput),
        "GetMenu": (app_models.GetMenuInput, app_models.GetMenuOutput),
        "PlaceOrder": (app_models.PlaceOrderInput, app_models.PlaceOrderOutput),
        "GetOrderHistory": (app_models.GetOrderHistoryInput, app_models.GetOrderHistoryOutput),
    }

    def _initialize_state(self) -> None:
        self.state = {
            "user_id": self.user_id,
            "restaurant_search_history": [],
            "order_history": [],
            "viewed_restaurants": [],
        }

    def _update_state(self, event, request_payload, response_payload) -> None:
        api_name = event.get("api_name", "")
        timestamp = event.get("timestamp", "")
        if api_name == "SearchRestaurants":
            query = request_payload.get("query")
            cuisine_type = request_payload.get("cuisine_type")
            if query or cuisine_type:
                search_record = {
                    "query": query,
                    "cuisine_type": cuisine_type,
                    "searched_at": timestamp,
                }
                self._append_to_state_history("restaurant_search_history", search_record, MAX_HISTORY_SMALL)
        elif api_name == "PlaceOrder":
            order = response_payload.get("order", {})
            if order:
                self._append_to_state_history("order_history", order, MAX_HISTORY_MEDIUM)
        elif api_name == "GetMenu":
            restaurant_id = request_payload.get("restaurant_id", "")
            if restaurant_id:
                viewed = self.state.setdefault("viewed_restaurants", [])
                if restaurant_id not in viewed:
                    self._append_to_state_history("viewed_restaurants", restaurant_id, MAX_HISTORY_MEDIUM)


class AppRegistry:
    APP_CLASSES: Dict[str, type[BaseApp]] = {
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
        "Google Maps": GoogleMapsApp,
        "UberEats": UberEatsApp,
    }

    def __init__(self) -> None:
        self.apps: Dict[Tuple[str, str], BaseApp] = {}

    def get_app(self, app_name: str, user_id: str) -> BaseApp:
        key = (app_name, user_id)
        if key not in self.apps:
            app_class = self.APP_CLASSES.get(app_name)
            if not app_class:
                raise ValueError(f"Unknown app: {app_name}. Available apps: {list(self.APP_CLASSES.keys())}")
            self.apps[key] = app_class(user_id)
        return self.apps[key]

    @classmethod
    def get_available_apps(cls) -> list[str]:
        return list(cls.APP_CLASSES.keys())


def _build_api_schemas() -> Dict[str, Dict[str, Dict[str, Any]]]:
    schemas: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for app_name, app_class in AppRegistry.APP_CLASSES.items():
        app_schemas: Dict[str, Dict[str, Any]] = {}
        for api_name, (input_model, output_model) in app_class.API_MODELS.items():
            app_schemas[api_name] = {
                "input": _model_to_schema(input_model),
                "output": _model_to_schema(output_model),
            }
        schemas[app_name] = app_schemas
    return schemas


APP_API_SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = _build_api_schemas()


def get_api_input_output_models(app_name: str, api_name: str) -> Tuple[type | None, type | None]:
    app_class = AppRegistry.APP_CLASSES.get(app_name)
    if not app_class:
        return (None, None)
    return app_class.API_MODELS.get(api_name, (None, None))
