"""App catalog used by the event chain prompt."""

APP_CATALOG = {
    "apps": [
        {
            "app_name": "Amazon",
            "app_category": "E-commerce",
            "relevant_life_domains": [
                "Finances & Material Living",
                "Leisure & Media Consumption",
            ],
            "apis": [
                {
                    "api_name": "SearchProducts",
                    "description": "Search for products on Amazon using keywords",
                    "selection_hint": [
                        "Shopping needs and product interests",
                        "Price sensitivity through search filters",
                        "Product category preferences",
                        "Purchase planning behavior",
                    ],
                },
                {
                    "api_name": "ShowProduct",
                    "description": (
                        "View detailed information about a specific product including price, "
                        "ratings, reviews, and description"
                    ),
                    "selection_hint": [
                        "Purchase decision-making process",
                        "Comparison shopping behavior",
                        "Quality consciousness through review reading",
                        "Time spent evaluating options",
                    ],
                },
                {
                    "api_name": "AddToCart",
                    "description": "Add a product to the shopping cart",
                    "selection_hint": [
                        "Purchase intent and immediacy",
                        "Impulsive vs. planned buying behavior",
                        "Shopping cart abandonment patterns",
                        "Multi-item purchasing habits",
                    ],
                },
                {
                    "api_name": "ShowCart",
                    "description": "View all items currently in the shopping cart",
                    "selection_hint": [
                        "Short-term purchase intentions",
                        "Cart management habits",
                        "Price threshold for checkout",
                        "Multi-session shopping behavior",
                    ],
                    "frequency": "medium",
                },
                {
                    "api_name": "ShowWishlist",
                    "description": (
                        "View all items saved in the wishlist for future consideration"
                    ),
                    "selection_hint": [
                        "Long-term purchase aspirations",
                        "Price monitoring behavior",
                        "Gift planning and special occasion preparation",
                        "Delayed gratification patterns",
                    ],
                },
                {
                    "api_name": "Checkout",
                    "description": "Complete the purchase of items in the cart",
                    "selection_hint": [
                        "Actual purchasing power and spending",
                        "Buying frequency and volume",
                        "Prime membership utilization",
                        "Payment method preferences",
                    ],
                },
            ],
        },
        {
            "app_name": "Spotify",
            "app_category": "Music Streaming",
            "relevant_life_domains": ["Leisure & Media Consumption"],
            "apis": [
                {
                    "api_name": "SearchSongs",
                    "description": "Search for songs, artists, or albums",
                    "selection_hint": [
                        "Music discovery behavior",
                        "Genre preferences and diversity",
                        "Openness to new artists",
                        "Music taste evolution",
                    ],
                },
                {
                    "api_name": "PlaySong",
                    "description": "Play a specific song and track listening duration",
                    "selection_hint": [
                        "Core music preferences and listening patterns",
                        "Daily routine and activity timing (workout music, commute, sleep)",
                        "Mood and emotional states",
                        "Song repetition and attachment behavior",
                        "Premium subscription status for ad-free experience",
                    ],
                },
                {
                    "api_name": "AddToPlaylist",
                    "description": "Add a song to a specific playlist",
                    "selection_hint": [
                        "Music curation and organization skills",
                        "Long-term music preferences",
                        "Playlist themes and life contexts (workout, study, party)",
                        "Collection-building behavior",
                    ],
                },
                {
                    "api_name": "FollowArtist",
                    "description": (
                        "Follow an artist to receive updates and recommendations"
                    ),
                    "selection_hint": [
                        "Artist loyalty and fandom intensity",
                        "Music taste identity and expression",
                        "Social signaling through artist choices",
                        "Engagement with music community",
                    ],
                },
            ],
        },
        {
            "app_name": "Fitbit",
            "app_category": "Health & Fitness Tracking",
            "relevant_life_domains": ["Health & Self-care"],
            "apis": [
                {
                    "api_name": "LogWorkout",
                    "description": (
                        "Manually log a workout session with type, duration, intensity, and location"
                    ),
                    "selection_hint": [
                        "Active fitness engagement and discipline",
                        "Preferred exercise types and variety",
                        "Workout intensity preferences",
                        "Workout locations (gym, home, park, etc.)",
                        "Self-tracking motivation and consistency",
                    ],
                },
                {
                    "api_name": "RecordActivity",
                    "description": (
                        "Record an outdoor activity with route and location tracking "
                        "(e.g., walking, hiking, outdoor run)"
                    ),
                    "selection_hint": [
                        "Outdoor activity preferences and habits",
                        "Favorite routes and locations for exercise",
                        "Distance and endurance patterns",
                        "Location-based activity patterns",
                    ],
                },
                {
                    "api_name": "SyncDevice",
                    "description": (
                        "Sync wearable device data including steps, heart rate, sleep patterns, "
                        "and passive activity"
                    ),
                    "selection_hint": [
                        "Daily activity levels and sedentary behavior",
                        "Sleep quality and schedule regularity",
                        "Cardiovascular health awareness",
                        "Technology adoption for health monitoring",
                    ],
                },
                {
                    "api_name": "SetGoals",
                    "description": (
                        "Set or update fitness goals such as daily steps, active minutes, "
                        "or weight targets"
                    ),
                    "selection_hint": [
                        "Health ambitions and self-expectations",
                        "Goal-setting realism vs. optimism",
                        "Commitment to lifestyle changes",
                        "Self-improvement priorities",
                    ],
                },
            ],
        },
        {
            "app_name": "Chase",
            "app_category": "Banking & Financial Management",
            "relevant_life_domains": ["Finances & Material Living"],
            "apis": [
                {
                    "api_name": "GetBalance",
                    "description": "Check current account balance",
                    "selection_hint": [
                        "Financial awareness and monitoring frequency",
                        "Money anxiety or security levels",
                        "Account checking habits as stress indicator",
                        "Financial buffer comfort zone",
                    ],
                    "frequency": "high",
                },
                {
                    "api_name": "GetTransactions",
                    "description": (
                        "View recent transaction history with merchant details, amounts, "
                        "and categories"
                    ),
                    "selection_hint": [
                        "Spending patterns across categories (dining, shopping, transportation)",
                        "Financial responsibility and tracking behavior",
                        "Lifestyle spending priorities",
                        "Cash flow management awareness",
                    ],
                    "frequency": "medium",
                },
                {
                    "api_name": "SearchTransactions",
                    "description": (
                        "Search for specific transactions by merchant, amount, or date range"
                    ),
                    "selection_hint": [
                        "Active financial management and recordkeeping",
                        "Expense dispute or verification needs",
                        "Tax preparation or budgeting diligence",
                        "Financial organization skills",
                    ],
                    "frequency": "low",
                },
                {
                    "api_name": "TransferMoney",
                    "description": "Transfer funds between accounts or to other people",
                    "selection_hint": [
                        "Liquidity management strategies",
                        "Savings discipline and allocation",
                        "Financial support relationships (family, friends)",
                        "Multi-account optimization behavior",
                    ],
                },
                {
                    "api_name": "PayBill",
                    "description": "Pay bills such as utilities, credit cards, or subscriptions",
                    "selection_hint": [
                        "Financial responsibility and payment timeliness",
                        "Recurring expense patterns",
                        "Bill management automation preferences",
                        "Essential vs. discretionary spending balance",
                    ],
                },
            ],
        },
        {
            "app_name": "Robinhood",
            "app_category": "Investment & Trading",
            "relevant_life_domains": ["Finances & Material Living"],
            "apis": [
                {
                    "api_name": "GetPortfolio",
                    "description": (
                        "View current investment holdings, positions, and portfolio value"
                    ),
                    "selection_hint": [
                        "Investment style (aggressive vs. conservative)",
                        "Asset diversification sophistication",
                        "Portfolio monitoring frequency and anxiety",
                        "Wealth accumulation and investment commitment",
                    ],
                },
                {
                    "api_name": "GetWatchlist",
                    "description": (
                        "View list of stocks or crypto being monitored for potential investment"
                    ),
                    "selection_hint": [
                        "Investment research and planning behavior",
                        "Market sector interests",
                        "Risk appetite indicators through watchlist choices",
                        "Patient vs. impulsive investing approach",
                    ],
                },
                {
                    "api_name": "SearchStocks",
                    "description": (
                        "Search for stocks or crypto by symbol or company name"
                    ),
                    "selection_hint": [
                        "Active investment research intensity",
                        "Market opportunity exploration",
                        "Financial curiosity and learning engagement",
                        "New investment consideration frequency",
                    ],
                },
                {
                    "api_name": "GetStockQuote",
                    "description": (
                        "View current price, change, and details for a specific stock or crypto"
                    ),
                    "selection_hint": [
                        "Market monitoring habits and timing",
                        "Price sensitivity and entry point strategy",
                        "Information-seeking before decisions",
                        "Investment due diligence thoroughness",
                    ],
                },
                {
                    "api_name": "BuyStock",
                    "description": "Execute a purchase of stocks or crypto",
                    "selection_hint": [
                        "Investment decision-making confidence",
                        "Capital deployment aggressiveness",
                        "Market timing beliefs and behavior",
                        "Financial risk tolerance in action",
                    ],
                },
                {
                    "api_name": "SellStock",
                    "description": "Execute a sale of stocks or crypto",
                    "selection_hint": [
                        "Profit-taking vs. loss-cutting discipline",
                        "Emotional response to market volatility",
                        "Exit strategy sophistication",
                        "Portfolio rebalancing awareness",
                    ],
                },
            ],
        },
        {
            "app_name": "WhatsApp",
            "app_category": "Instant Messaging",
            "relevant_life_domains": [
                "Family & Close Relationships",
                "Social & Community",
            ],
            "apis": [
                {
                    "api_name": "GetMessages",
                    "description": (
                        "Retrieve message history from a specific contact or group"
                    ),
                    "selection_hint": [
                        "Communication frequency with different relationships",
                        "Relationship intimacy and depth through message volume",
                        "Conversation review and reminiscence behavior",
                        "Social network structure and priority contacts",
                    ],
                },
                {
                    "api_name": "SendMessage",
                    "description": "Send a text message to a contact or group",
                    "selection_hint": [
                        "Communication initiation patterns",
                        "Message length and conversation depth preferences",
                        "Response speed and availability signals",
                        "Relationship maintenance effort and priorities",
                    ],
                },
                {
                    "api_name": "SendMedia",
                    "description": "Send photos, videos, or voice messages",
                    "selection_hint": [
                        "Rich communication preferences",
                        "Life moment sharing behavior",
                        "Visual vs. text communication style",
                        "Intimacy expression through media types",
                    ],
                },
            ],
        },
        {
            "app_name": "Gmail",
            "app_category": "Email",
            "relevant_life_domains": ["Work & Education", "Social & Community"],
            "apis": [
                {
                    "api_name": "GetInbox",
                    "description": (
                        "Retrieve current inbox emails with previews and metadata"
                    ),
                    "selection_hint": [
                        "Email volume as work intensity indicator",
                        "Inbox management style (inbox zero vs. accumulator)",
                        "Information overload levels",
                        "Professional communication burden",
                    ],
                    "frequency": "very_high",
                },
                {
                    "api_name": "ReadEmail",
                    "description": "Open and read a specific email",
                    "selection_hint": [
                        "Email prioritization and triage decisions",
                        "Information processing speed",
                        "Attention allocation to different senders",
                        "Email response time patterns",
                    ],
                },
                {
                    "api_name": "SendEmail",
                    "description": "Compose and send a new email",
                    "selection_hint": [
                        "Proactive communication and initiative",
                        "Professional relationship building",
                        "Email formality and communication style",
                        "Work productivity and output generation",
                    ],
                },
                {
                    "api_name": "ReplyEmail",
                    "description": "Reply to a received email",
                    "selection_hint": [
                        "Responsiveness and reliability",
                        "Communication reciprocity patterns",
                        "Reply speed by sender relationship",
                        "Professional courtesy and engagement",
                    ],
                },
            ],
        },
        {
            "app_name": "LinkedIn",
            "app_category": "Professional Networking",
            "relevant_life_domains": ["Work & Education", "Social & Community"],
            "apis": [
                {
                    "api_name": "UpdateProfile",
                    "description": (
                        "Update profile information such as headline, summary, or photo"
                    ),
                    "selection_hint": [
                        "Personal branding awareness and effort",
                        "Career positioning and messaging",
                        "Professional identity evolution",
                        "Job market readiness signals",
                    ],
                },
                {
                    "api_name": "AddExperience",
                    "description": "Add or update work experience entries",
                    "selection_hint": [
                        "Career progression and mobility",
                        "Achievement documentation habits",
                        "Professional milestone celebration",
                        "Resume maintenance discipline",
                    ],
                },
                {
                    "api_name": "AddSkill",
                    "description": "Add new skills to profile",
                    "selection_hint": [
                        "Skill development and learning focus areas",
                        "Career development strategy",
                        "Professional growth mindset",
                        "Market positioning through skill signals",
                    ],
                },
                {
                    "api_name": "PostUpdate",
                    "description": "Share a post, article, or thought on LinkedIn feed",
                    "selection_hint": [
                        "Thought leadership aspirations",
                        "Professional content creation and sharing",
                        "Industry engagement and visibility efforts",
                        "Personal brand building activity",
                    ],
                },
                {
                    "api_name": "GetFeed",
                    "description": (
                        "View LinkedIn feed with posts from connections and followed pages"
                    ),
                    "selection_hint": [
                        "Professional content consumption habits",
                        "Industry news and trend awareness",
                        "Learning and development engagement",
                        "Professional network monitoring",
                    ],
                },
                {
                    "api_name": "LikePost",
                    "description": "Like a post in the feed",
                    "selection_hint": [
                        "Content preference signals",
                        "Network engagement and support behavior",
                        "Professional relationship nurturing",
                        "Visibility and presence maintenance",
                    ],
                },
                {
                    "api_name": "CommentOnPost",
                    "description": (
                        "Comment on a post to share thoughts or engage in discussion"
                    ),
                    "selection_hint": [
                        "Deep engagement with professional content",
                        "Thought leadership and expertise demonstration",
                        "Network relationship deepening efforts",
                        "Discussion participation willingness",
                    ],
                },
                {
                    "api_name": "SearchJobs",
                    "description": (
                        "Search for job openings by keywords, location, or company"
                    ),
                    "selection_hint": [
                        "Active job seeking status and intensity",
                        "Career change considerations",
                        "Job market exploration and dissatisfaction signals",
                        "Career goals and aspirations",
                    ],
                },
                {
                    "api_name": "ApplyJob",
                    "description": "Submit application for a job posting",
                    "selection_hint": [
                        "Serious job transition intent",
                        "Job application volume and selectivity",
                        "Career change readiness",
                        "Job search commitment level",
                    ],
                },
                {
                    "api_name": "SendConnectionRequest",
                    "description": (
                        "Send a connection request to another LinkedIn user"
                    ),
                    "selection_hint": [
                        "Networking proactivity and strategy",
                        "Professional relationship building efforts",
                        "Career network expansion goals",
                        "Social capital investment behavior",
                    ],
                },
            ],
        },
        {
            "app_name": "Notion",
            "app_category": "Knowledge Management & Productivity",
            "relevant_life_domains": ["Work & Education", "Health & Self-care"],
            "apis": [
                {
                    "api_name": "GetPages",
                    "description": "Retrieve list of pages and notebooks in workspace",
                    "selection_hint": [
                        "Knowledge management system scope",
                        "Organization complexity and structure",
                        "Content creation volume and diversity",
                        "Digital workspace organization style",
                    ],
                },
                {
                    "api_name": "CreatePage",
                    "description": "Create a new page or note",
                    "selection_hint": [
                        "Knowledge production and documentation habits",
                        "Note-taking frequency and triggers",
                        "Thinking and learning process externalization",
                        "Creative or analytical work patterns",
                    ],
                },
                {
                    "api_name": "UpdatePage",
                    "description": "Edit and update existing page content",
                    "selection_hint": [
                        "Iterative thinking and refinement behavior",
                        "Content maintenance and quality standards",
                        "Knowledge evolution and updates tracking",
                        "Perfectionism vs. completion tendencies",
                    ],
                },
                {
                    "api_name": "SearchContent",
                    "description": "Search across all pages and databases",
                    "selection_hint": [
                        "Information retrieval efficiency needs",
                        "Knowledge reuse and reference behavior",
                        "Memory reliance vs. search dependence",
                        "Information organization effectiveness",
                    ],
                },
                {
                    "api_name": "CreateDatabaseEntry",
                    "description": (
                        "Add entry to a database (task, project, habit tracker, etc.)"
                    ),
                    "selection_hint": [
                        "Structured productivity and tracking systems",
                        "Task and project management discipline",
                        "Quantified self and habit tracking behavior",
                        "Goal-oriented planning and execution",
                    ],
                },
            ],
        },
        {
            "app_name": "Netflix",
            "app_category": "Video Streaming",
            "relevant_life_domains": ["Leisure & Media Consumption"],
            "apis": [
                {
                    "api_name": "SearchContent",
                    "description": "Search for movies, TV shows, or documentaries",
                    "selection_hint": [
                        "Active content discovery preferences",
                        "Genre and topic interests",
                        "Specific viewing intent vs. browsing",
                        "Decision-making approach for entertainment",
                    ],
                    "frequency": "medium",
                },
                {
                    "api_name": "ShowTitle",
                    "description": (
                        "View detailed information about a specific title including description, "
                        "cast, and ratings"
                    ),
                    "selection_hint": [
                        "Content evaluation thoroughness",
                        "Decision-making deliberation for viewing",
                        "Quality consciousness and selectivity",
                        "Time spent on content selection",
                    ],
                },
                {
                    "api_name": "PlayContent",
                    "description": "Start playing a movie or TV show episode",
                    "selection_hint": [
                        "Viewing frequency and binge-watching patterns",
                        "Content preferences and genre tastes",
                        "Viewing time distribution (weekday vs. weekend, time of day)",
                        "Watch duration and completion rates",
                        "Subscription tier for streaming quality",
                    ],
                },
                {
                    "api_name": "AddToMyList",
                    "description": "Add a title to personal watchlist",
                    "selection_hint": [
                        "Content curation and planning behavior",
                        "Delayed viewing intentions",
                        "Aspiration vs. actual viewing gap",
                        "List management and follow-through",
                    ],
                },
                {
                    "api_name": "RateContent",
                    "description": "Rate a watched title with thumbs up or down",
                    "selection_hint": [
                        "Feedback and opinion expression willingness",
                        "Algorithm training engagement",
                        "Content evaluation standards and taste clarity",
                        "Platform interaction and investment",
                    ],
                },
            ],
        },
        {
            "app_name": "Goodreads",
            "app_category": "Book Tracking & Reviews",
            "relevant_life_domains": [
                "Leisure & Media Consumption",
                "Work & Education",
            ],
            "apis": [
                {
                    "api_name": "SearchBooks",
                    "description": "Search for books by title, author, or keywords",
                    "selection_hint": [
                        "Reading interests and topic preferences",
                        "Book discovery methods (recommendations vs. direct search)",
                        "Genre preferences and reading diversity",
                        "Intellectual curiosity areas",
                    ],
                },
                {
                    "api_name": "ShowBook",
                    "description": (
                        "View detailed information about a specific book including synopsis, "
                        "ratings, and reviews"
                    ),
                    "selection_hint": [
                        "Reading decision-making thoroughness",
                        "Book selection criteria and standards",
                        "Review reliance and opinion-seeking",
                        "Quality consciousness for reading material",
                    ],
                },
                {
                    "api_name": "AddToShelf",
                    "description": (
                        "Add a book to a specific shelf (want-to-read, currently-reading, read)"
                    ),
                    "selection_hint": [
                        "Reading planning and intention setting",
                        "Book collection curation behavior",
                        "Reading progress tracking discipline",
                        "Aspirational vs. actual reading habits",
                    ],
                },
                {
                    "api_name": "RateBook",
                    "description": "Rate a book on a 1-5 star scale",
                    "selection_hint": [
                        "Reading engagement and completion",
                        "Critical thinking and evaluation skills",
                        "Rating standards and generosity",
                        "Personal taste clarity and confidence",
                    ],
                    "frequency": "low",
                },
                {
                    "api_name": "WriteReview",
                    "description": "Write a text review for a book",
                    "selection_hint": [
                        "Deep reflection on reading experience",
                        "Written expression and articulation skills",
                        "Willingness to share opinions publicly",
                        "Intellectual engagement depth with material",
                    ],
                    "frequency": "very_low",
                },
            ],
        },
        {
            "app_name": "Instagram",
            "app_category": "Social Media & Photo Sharing",
            "relevant_life_domains": [
                "Social & Community",
                "Leisure & Media Consumption",
            ],
            "apis": [
                {
                    "api_name": "CreatePost",
                    "description": (
                        "Create a new photo or video post with caption and optional location tag"
                    ),
                    "selection_hint": [
                        "Content creation frequency and style",
                        "Location sharing behavior and privacy preferences",
                        "Life moments and experiences worth sharing",
                        "Personal branding and social identity expression",
                    ],
                },
                {
                    "api_name": "PostStory",
                    "description": (
                        "Post a photo or video to Instagram Stories (24-hour temporary content) "
                        "with optional location tag"
                    ),
                    "selection_hint": [
                        "Daily life sharing frequency and openness",
                        "Casual vs. curated content preferences",
                        "Social presence maintenance",
                        "Ephemeral vs. permanent sharing comfort",
                        "Location sharing in real-time",
                    ],
                },
                {
                    "api_name": "LikePost",
                    "description": "Like a post in the feed",
                    "selection_hint": [
                        "Social engagement levels and generosity",
                        "Content consumption patterns and interests",
                        "Relationship acknowledgment behavior",
                        "Feed scrolling depth and time",
                    ],
                },
                {
                    "api_name": "CommentOnPost",
                    "description": "Comment on a post",
                    "selection_hint": [
                        "Deep social engagement willingness",
                        "Relationship investment and maintenance",
                        "Public communication comfort",
                        "Thoughtfulness in interactions",
                    ],
                },
                {
                    "api_name": "SendDirectMessage",
                    "description": (
                        "Send a private message to another user"
                    ),
                    "selection_hint": [
                        "Private communication preferences",
                        "Content sharing behavior (memes, posts, personal messages)",
                        "Close friendship maintenance",
                        "Social initiation patterns",
                    ],
                },
                {
                    "api_name": "FollowUser",
                    "description": "Follow another user's account",
                    "selection_hint": [
                        "Social network expansion behavior",
                        "Interest-based following vs. social obligation",
                        "Content curation preferences",
                        "New relationship openness",
                    ],
                },
                {
                    "api_name": "UnfollowUser",
                    "description": "Unfollow a user's account",
                    "selection_hint": [
                        "Social network curation and pruning",
                        "Relationship ending or distancing",
                        "Content quality standards enforcement",
                        "Digital boundary setting",
                    ],
                },
                {
                    "api_name": "GetFollowing",
                    "description": "View list of accounts currently followed",
                    "selection_hint": [
                        "Social network size and composition review",
                        "Following audit and cleanup consideration",
                        "Social comparison behavior",
                        "Network management awareness",
                    ],
                },
            ],
        },
        {
            "app_name": "Google",
            "app_category": "Search Engine",
            "relevant_life_domains": ["All domains - cross-cutting tool"],
            "apis": [
                {
                    "api_name": "Search",
                    "description": "Perform a web search and receive list of results",
                    "selection_hint": [
                        "Information needs and curiosity areas",
                        "Search query formulation sophistication",
                        "Problem-solving approach (search vs. ask AI)",
                        "Fact-checking and verification habits",
                    ],
                },
                {
                    "api_name": "ClickResult",
                    "description": (
                        "Click on a specific search result to view the webpage"
                    ),
                    "selection_hint": [
                        "Result evaluation and selection criteria",
                        "Source trustworthiness judgment",
                        "Information gathering depth",
                        "Click position bias (top results vs. deeper exploration)",
                    ],
                },
            ],
        },
        {
            "app_name": "LLM Assistant",
            "app_category": "AI Assistant",
            "relevant_life_domains": ["All domains - cross-cutting tool"],
            "apis": [
                {
                    "api_name": "CreateConversation",
                    "description": "Start a new conversation thread with the AI assistant",
                    "selection_hint": [
                        "Task switching and compartmentalization",
                        "New problem or topic initiation",
                        "AI usage frequency and dependency",
                        "Conversation organization preferences",
                    ],
                },
                {
                    "api_name": "ContinueConversation",
                    "description": (
                        "Send a message in an existing conversation thread"
                    ),
                    "selection_hint": [
                        "Conversation continuity and depth",
                        "Complex task breakdown and iteration",
                        "Clarification and refinement patterns",
                        "Multi-turn interaction engagement",
                        "Query types and domains (work, learning, creative, personal)",
                        "Problem-solving approach and follow-through",
                    ],
                },
            ],
        },
        {
            "app_name": "Google Maps",
            "app_category": "Navigation & Location Services",
            "relevant_life_domains": [
                "Social & Community",
                "Leisure & Media Consumption",
                "Health & Self-care",
            ],
            "apis": [
                {
                    "api_name": "ShareLocation",
                    "description": (
                        "Share current location with another person for a specified duration"
                    ),
                    "selection_hint": [
                        "Trust and relationship closeness indicators",
                        "Safety consciousness and coordination habits",
                        "Privacy comfort levels with location sharing",
                        "Social coordination patterns",
                    ],
                },
                {
                    "api_name": "CheckIn",
                    "description": (
                        "Check in at a specific location to record presence"
                    ),
                    "selection_hint": [
                        "Places visited and lifestyle patterns",
                        "Social activity and venue preferences",
                        "Location-based habits and routines",
                        "Experience documentation behavior",
                    ],
                },
                {
                    "api_name": "GetDirections",
                    "description": (
                        "Get directions between two locations with travel mode options"
                    ),
                    "selection_hint": [
                        "Transportation preferences and habits",
                        "Commute patterns and frequently visited places",
                        "Travel planning behavior",
                        "Environmental consciousness through travel mode choices",
                    ],
                },
                {
                    "api_name": "SearchPlaces",
                    "description": (
                        "Search for nearby places by category or keyword"
                    ),
                    "selection_hint": [
                        "Local exploration and discovery behavior",
                        "Venue preferences and interests",
                        "Spontaneous vs. planned activity patterns",
                        "Location-based decision making",
                    ],
                },
            ],
        },
        {
            "app_name": "UberEats",
            "app_category": "Food Delivery",
            "relevant_life_domains": [
                "Finances & Material Living",
                "Health & Self-care",
            ],
            "apis": [
                {
                    "api_name": "SearchRestaurants",
                    "description": (
                        "Search for restaurants by name, cuisine type, or category"
                    ),
                    "selection_hint": [
                        "Food preferences and dietary habits",
                        "Cuisine diversity and exploration",
                        "Restaurant discovery behavior",
                        "Local dining knowledge",
                    ],
                },
                {
                    "api_name": "GetMenu",
                    "description": (
                        "View menu items and prices for a specific restaurant"
                    ),
                    "selection_hint": [
                        "Menu browsing and selection behavior",
                        "Price sensitivity for food delivery",
                        "Dietary restrictions and preferences",
                        "Decision-making process for food orders",
                    ],
                },
                {
                    "api_name": "PlaceOrder",
                    "description": (
                        "Place a food delivery order with items and delivery address"
                    ),
                    "selection_hint": [
                        "Food delivery frequency and spending",
                        "Meal ordering patterns (solo vs. group, meal types)",
                        "Tipping behavior and generosity",
                        "Delivery address reveals home/work locations",
                        "Cooking vs. ordering out balance",
                    ],
                },
                {
                    "api_name": "GetOrderHistory",
                    "description": (
                        "View past food delivery orders"
                    ),
                    "selection_hint": [
                        "Favorite restaurants and repeat orders",
                        "Order frequency patterns",
                        "Spending tracking behavior",
                        "Dietary consistency or variety",
                    ],
                },
            ],
        },
    ]
}
