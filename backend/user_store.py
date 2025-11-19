# ============================================================
#           USER PROFILE IN-MEMORY STORE
#           Simple session-based user profile management
# ============================================================

import json
from typing import Optional, Dict
from datetime import datetime

# In-memory store for user profiles
# Key: session_id, Value: user profile dict
USER_PROFILES: Dict[str, dict] = {}


def load_profile(session_id: Optional[str] = None) -> dict:
    """
    Load user profile from in-memory store.
    Returns default profile if session_id not found.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        User profile dict with preferred_tone, name, history_summary
    """
    if not session_id:
        return get_default_profile()
    
    # Check if profile exists in store
    if session_id in USER_PROFILES:
        profile = USER_PROFILES[session_id]
        # Update last_accessed timestamp
        profile['last_accessed'] = datetime.now().isoformat()
        return profile
    
    # Create new profile for this session
    new_profile = get_default_profile()
    new_profile['session_id'] = session_id
    new_profile['created_at'] = datetime.now().isoformat()
    new_profile['last_accessed'] = datetime.now().isoformat()
    
    USER_PROFILES[session_id] = new_profile
    return new_profile


def save_profile(session_id: str, profile: dict) -> bool:
    """
    Save user profile to in-memory store.
    
    Args:
        session_id: Unique session identifier
        profile: User profile dict to save
        
    Returns:
        True if successful
    """
    try:
        profile['session_id'] = session_id
        profile['updated_at'] = datetime.now().isoformat()
        USER_PROFILES[session_id] = profile
        return True
    except Exception as e:
        print(f"Error saving profile: {e}")
        return False


def update_profile_field(session_id: str, field: str, value: any) -> bool:
    """
    Update a specific field in user profile.
    
    Args:
        session_id: Unique session identifier
        field: Field name to update
        value: New value for the field
        
    Returns:
        True if successful
    """
    try:
        profile = load_profile(session_id)
        profile[field] = value
        return save_profile(session_id, profile)
    except Exception as e:
        print(f"Error updating profile field: {e}")
        return False


def delete_profile(session_id: str) -> bool:
    """
    Delete user profile from store.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        True if profile was deleted
    """
    try:
        if session_id in USER_PROFILES:
            del USER_PROFILES[session_id]
            return True
        return False
    except Exception as e:
        print(f"Error deleting profile: {e}")
        return False


def get_default_profile() -> dict:
    """
    Get default user profile structure.
    
    Returns:
        Default profile dict
    """
    return {
        "preferred_tone": "warm_personal",
        "name": "",
        "history_summary": "",
        "conversation_count": 0,
        "health_concerns": [],
        "preferences": {
            "voice_enabled": True,
            "detailed_explanations": True,
            "emoji_enabled": True,
        }
    }


def get_all_profiles() -> Dict[str, dict]:
    """
    Get all user profiles (for admin purposes).
    
    Returns:
        Dictionary of all profiles
    """
    return USER_PROFILES.copy()


def cleanup_old_sessions(max_age_hours: int = 24) -> int:
    """
    Remove profiles that haven't been accessed in max_age_hours.
    
    Args:
        max_age_hours: Maximum age in hours for inactive sessions
        
    Returns:
        Number of profiles deleted
    """
    from datetime import timedelta
    
    deleted_count = 0
    current_time = datetime.now()
    sessions_to_delete = []
    
    for session_id, profile in USER_PROFILES.items():
        if 'last_accessed' in profile:
            try:
                last_accessed = datetime.fromisoformat(profile['last_accessed'])
                age = current_time - last_accessed
                
                if age > timedelta(hours=max_age_hours):
                    sessions_to_delete.append(session_id)
            except Exception as e:
                print(f"Error checking session age: {e}")
    
    for session_id in sessions_to_delete:
        if delete_profile(session_id):
            deleted_count += 1
    
    return deleted_count
