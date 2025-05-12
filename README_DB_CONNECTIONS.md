# Database Connection Management

This document describes the standardized approach to database connections in the CortexFlow project.

## Context Manager Pattern

We've implemented a context manager pattern for database connections in the `KnowledgeStore` class to ensure consistent handling of connections:

```python
@contextmanager
def get_connection(self) -> sqlite3.Connection:
    """
    Context manager for database connections.
    
    Returns a connection to the database, either the persistent in-memory connection
    or a new connection to the file-based database if in-memory is not available.
    
    The connection is automatically closed if it's a new connection.
    """
    if self.conn is not None:
        # Using in-memory database
        yield self.conn
    else:
        # Create a new connection to the file-based database
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
```

## Usage

Instead of manually creating and closing connections in each method, we now use the context manager:

```python
# Before:
if self.conn is not None:
    # Using in-memory database
    conn = self.conn
else:
    # Using file-based database
    conn = sqlite3.connect(self.db_path)

cursor = conn.cursor()
# ... operations ...

if self.conn is None:
    conn.close()
```

```python
# After:
with self.get_connection() as conn:
    cursor = conn.cursor()
    # ... operations ...
```

## Benefits

1. **Consistency**: All database operations use the same connection pattern
2. **Safety**: Connections are properly closed, even if exceptions occur
3. **Readability**: Code is cleaner and more concise
4. **Maintainability**: Connection management logic is centralized in one place

## Persistent vs. Temporary Connections

The context manager handles two cases:
- Using an existing persistent in-memory connection (`self.conn`)
- Creating a temporary file-based connection when needed

This allows the code to work efficiently with either approach. 