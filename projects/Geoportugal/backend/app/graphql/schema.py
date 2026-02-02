import strawberry

from app.graphql.resolvers import Query

schema = strawberry.Schema(query=Query)
