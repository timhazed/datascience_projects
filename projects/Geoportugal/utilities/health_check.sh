#!/bin/bash

# =============================================================================
# GeoPortugal Health Check Script
# =============================================================================
# Tests backend API, GraphQL, and frontend services with known data queries
# Reports comprehensive status of all system components
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================

# Service endpoints
BACKEND_HOST="localhost"
BACKEND_PORT="8000"
FRONTEND_HOST="localhost"  
FRONTEND_PORT="3000"
GRAPHQL_ENDPOINT="http://${BACKEND_HOST}:${BACKEND_PORT}/graphql"
REST_API_ENDPOINT="http://${BACKEND_HOST}:${BACKEND_PORT}/api/v1"
FRONTEND_ENDPOINT="http://${FRONTEND_HOST}:${FRONTEND_PORT}"

# Test timeouts (seconds)
TIMEOUT=10
CONNECTION_TIMEOUT=5

# Known test data (adjust based on your loaded data)
TEST_SEARCH_TERM="Porto"
TEST_SEARCH_TERM_ENGLISH="Lisbon"
TEST_SEARCH_TERM_ACCENT="Óbidos"
TEST_CITY_SEARCH="Lisboa"
TEST_NEARBY_LAT="38.7223"
TEST_NEARBY_LNG="-9.1393"
EXPECTED_MIN_RESULTS=5
EXPECTED_CITY_RESULTS=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_header() {
    echo -e "\n${CYAN}=================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}=================================================${NC}"
}

print_section() {
    echo -e "\n${BLUE}🔍 $1${NC}"
    echo -e "${BLUE}$(echo "$1" | sed 's/./─/g')${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}ℹ️  $1${NC}"
}

# Check if a service is responding on a specific port
check_port() {
    local host=$1
    local port=$2
    local service_name=$3
    
    # Use curl to check HTTP connectivity instead of raw TCP
    if curl -s --max-time $CONNECTION_TIMEOUT "http://$host:$port" >/dev/null 2>&1; then
        print_success "$service_name is responding on $host:$port"
        return 0
    else
        print_error "$service_name is not responding on $host:$port"
        return 1
    fi
}

# Make HTTP request with error handling
make_request() {
    local url=$1
    local method=${2:-GET}
    local data=$3
    local description=$4
    
    if [ -n "$data" ]; then
        response=$(curl -s -w "HTTPSTATUS:%{http_code}\nTIME:%{time_total}" \
                   --max-time $TIMEOUT \
                   -X "$method" \
                   -H "Content-Type: application/json" \
                   -d "$data" \
                   "$url" 2>/dev/null)
    else
        response=$(curl -s -w "HTTPSTATUS:%{http_code}\nTIME:%{time_total}" \
                   --max-time $TIMEOUT \
                   -X "$method" \
                   "$url" 2>/dev/null)
    fi
    
    if [ $? -eq 0 ]; then
        http_code=$(echo "$response" | grep "HTTPSTATUS:" | cut -d: -f2)
        response_time=$(echo "$response" | grep "TIME:" | cut -d: -f2)
        body=$(echo "$response" | sed '/HTTPSTATUS:/d' | sed '/TIME:/d')
        
        if [ "$http_code" = "200" ]; then
            print_success "$description (${response_time}s)"
            echo "$body"
            return 0
        else
            print_error "$description - HTTP $http_code (${response_time}s)"
            return 1
        fi
    else
        print_error "$description - Connection failed"
        return 1
    fi
}

# Safe JSON value extraction without jq dependency
extract_json_count() {
    local json_response="$1"
    local search_pattern="$2"
    
    # For GraphQL responses, look for the actual data structure
    if echo "$json_response" | grep -q "\"data\""; then
        # Count occurrences of the search pattern within the data object
        local count=$(echo "$json_response" | grep -o "\"$search_pattern\"" | wc -l 2>/dev/null | tr -d ' ')
    else
        # Fallback for non-GraphQL responses
        local count=$(echo "$json_response" | grep -o "\"$search_pattern\"" | wc -l 2>/dev/null | tr -d ' ')
    fi
    
    # Ensure count is numeric
    if ! [[ "$count" =~ ^[0-9]+$ ]]; then
        count=0
    fi
    
    echo "$count"
}

# Safe numeric comparison
safe_compare() {
    local value1="$1"
    local operator="$2" 
    local value2="$3"
    
    # Ensure both values are numeric
    if ! [[ "$value1" =~ ^[0-9]+$ ]]; then
        value1=0
    fi
    if ! [[ "$value2" =~ ^[0-9]+$ ]]; then
        value2=0
    fi
    
    case "$operator" in
        "-gt") [ "$value1" -gt "$value2" ] ;;
        "-ge") [ "$value1" -ge "$value2" ] ;;
        "-eq") [ "$value1" -eq "$value2" ] ;;
        "-lt") [ "$value1" -lt "$value2" ] ;;
        "-le") [ "$value1" -le "$value2" ] ;;
        *) return 1 ;;
    esac
}

# =============================================================================
# HEALTH CHECK FUNCTIONS
# =============================================================================

check_backend_health() {
    print_section "Backend API Health Checks"
    
    # Check basic connectivity
    if ! check_port "$BACKEND_HOST" "$BACKEND_PORT" "Backend API"; then
        return 1
    fi
    
    # Check health endpoint
    local health_response
    health_response=$(make_request "$REST_API_ENDPOINT/health" "GET" "" "Basic health check")
    if [ $? -eq 0 ]; then
        print_info "Health status: $(echo "$health_response" | jq -r '.status // "unknown"' 2>/dev/null || echo "OK")"
    fi
    
    # Check detailed health endpoint
    local detailed_health
    detailed_health=$(make_request "$REST_API_ENDPOINT/health/detailed" "GET" "" "Detailed health check")
    if [ $? -eq 0 ]; then
        print_info "Database: $(echo "$detailed_health" | jq -r '.database.status // "unknown"' 2>/dev/null || echo "Unknown")"
        print_info "Redis: $(echo "$detailed_health" | jq -r '.redis.status // "unknown"' 2>/dev/null || echo "Unknown")"
    fi
    
    # Check database connectivity (if health endpoints don't exist)
    check_database_connection
    
    return 0
}

check_database_connection() {
    print_info "Testing database connectivity..."
    
    # Test PostgreSQL connection (if available)
    if command -v psql >/dev/null 2>&1; then
        if PGPASSWORD=postgres psql -h localhost -U postgres -d geoportugal -c "SELECT 1;" >/dev/null 2>&1; then
            print_success "PostgreSQL database is accessible"
        else
            print_warning "Direct PostgreSQL connection failed (may be normal if using different credentials)"
        fi
    fi
    
    # Test Redis connection (if available)
    if command -v redis-cli >/dev/null 2>&1; then
        if redis-cli -h localhost ping 2>/dev/null | grep -q "PONG"; then
            print_success "Redis cache is accessible"
        else
            print_warning "Redis connection failed (may be normal if not running locally)"
        fi
    fi
}

check_rest_api() {
    print_section "REST API Data Tests"
    
    # Test districts endpoint
    local districts_response
    districts_response=$(make_request "$REST_API_ENDPOINT/districts?limit=5" "GET" "" "Districts API")
    if [ $? -eq 0 ]; then
        local district_count
        district_count=$(extract_json_count "$districts_response" "name")
        print_info "Districts returned: $district_count"
    fi
    
    # Test search endpoint
    local search_response
    search_response=$(make_request "$REST_API_ENDPOINT/search?q=$TEST_SEARCH_TERM&limit=10" "GET" "" "Search API")
    if [ $? -eq 0 ]; then
        local locality_count
        locality_count=$(extract_json_count "$search_response" "name")
        print_info "Search results for '$TEST_SEARCH_TERM': $locality_count localities"
        
        if safe_compare "$locality_count" "-ge" "$EXPECTED_MIN_RESULTS"; then
            print_success "Search returned sufficient results ($locality_count >= $EXPECTED_MIN_RESULTS)"
        else
            print_warning "Search returned fewer results than expected ($locality_count < $EXPECTED_MIN_RESULTS)"
        fi
    fi
    
    return 0
}

check_graphql() {
    print_section "GraphQL API Tests"
    
    # Test basic GraphQL introspection
    local introspection_query='{"query": "{ __schema { types { name } } }"}'
    local introspection_response
    introspection_response=$(make_request "$GRAPHQL_ENDPOINT" "POST" "$introspection_query" "GraphQL introspection")
    if [ $? -eq 0 ]; then
        local schema_types
        schema_types=$(extract_json_count "$introspection_response" "name")
        print_info "GraphQL schema types: $schema_types"
    fi
    
    # Test general search
    local general_query="{\"query\": \"{ searchLocations(query: \\\"$TEST_SEARCH_TERM\\\") { name featureType latitude longitude } }\"}"
    local general_response
    general_response=$(make_request "$GRAPHQL_ENDPOINT" "POST" "$general_query" "General search ($TEST_SEARCH_TERM)")
    if [ $? -eq 0 ]; then
        local result_count
        result_count=$(extract_json_count "$general_response" "name")
        print_info "General search results: $result_count"
        
        if safe_compare "$result_count" "-ge" "$EXPECTED_MIN_RESULTS"; then
            print_success "General search working correctly"
        else
            print_warning "General search returned fewer results than expected"
        fi
    fi
    
    # Test city search (new feature)
    local city_query="{\"query\": \"{ citySearch(query: \\\"$TEST_CITY_SEARCH\\\") { name featureType population latitude longitude } }\"}"
    local city_response
    city_response=$(make_request "$GRAPHQL_ENDPOINT" "POST" "$city_query" "City search ($TEST_CITY_SEARCH)")
    if [ $? -eq 0 ]; then
        local city_count
        city_count=$(extract_json_count "$city_response" "name")
        print_info "City search results: $city_count"
        
        if safe_compare "$city_count" "-ge" "$EXPECTED_CITY_RESULTS"; then
            print_success "City search working correctly (found capital)"
        else
            print_warning "City search may not be working - Lisboa not found"
        fi
    fi
    
    # Test English search (Lisbon vs Lisboa)
    local english_query="{\"query\": \"{ citySearch(query: \\\"$TEST_SEARCH_TERM_ENGLISH\\\") { name featureType latitude longitude } }\"}"
    local english_response
    english_response=$(make_request "$GRAPHQL_ENDPOINT" "POST" "$english_query" "English city search ($TEST_SEARCH_TERM_ENGLISH)")
    if [ $? -eq 0 ]; then
        local english_count
        english_count=$(extract_json_count "$english_response" "name")
        print_info "English search results: $english_count"
        
        if safe_compare "$english_count" "-gt" "0"; then
            print_success "English search enhancement working correctly"
        else
            print_warning "English search enhancement may not be working"
        fi
    fi
    
    # Test nearby locations feature
    local nearby_query="{\"query\": \"{ localitiesNear(lat: $TEST_NEARBY_LAT, lng: $TEST_NEARBY_LNG, radiusKm: 10.0) { name featureType latitude longitude } }\"}"
    local nearby_response
    nearby_response=$(make_request "$GRAPHQL_ENDPOINT" "POST" "$nearby_query" "Nearby locations search")
    if [ $? -eq 0 ]; then
        local nearby_count
        nearby_count=$(extract_json_count "$nearby_response" "name")
        print_info "Nearby locations results: $nearby_count"
        
        if safe_compare "$nearby_count" "-gt" "5"; then
            print_success "Nearby locations search working correctly"
        else
            print_warning "Nearby locations search returned fewer results than expected"
        fi
    fi
    
    # Test accent-insensitive search
    local accent_query="{\"query\": \"{ searchLocations(query: \\\"$TEST_SEARCH_TERM_ACCENT\\\") { name featureType latitude longitude } }\"}"
    local accent_response
    accent_response=$(make_request "$GRAPHQL_ENDPOINT" "POST" "$accent_query" "Accent-insensitive search ($TEST_SEARCH_TERM_ACCENT)")
    if [ $? -eq 0 ]; then
        local accent_count
        accent_count=$(extract_json_count "$accent_response" "name")
        print_info "Accent-insensitive results: $accent_count"
        
        if safe_compare "$accent_count" "-gt" "0"; then
            print_success "Accent-insensitive search working correctly"
        else
            print_warning "Accent-insensitive search may not be working"
        fi
    fi
    
    return 0
}

check_frontend() {
    print_section "Frontend Application Tests"
    
    # Check basic connectivity
    if ! check_port "$FRONTEND_HOST" "$FRONTEND_PORT" "Frontend"; then
        return 1
    fi
    
    # Check if frontend is serving content
    local frontend_response
    frontend_response=$(make_request "$FRONTEND_ENDPOINT" "GET" "" "Frontend homepage")
    if [ $? -eq 0 ]; then
        # Check for React/Next.js indicators
        if echo "$frontend_response" | grep -q "React\|Next\|__NEXT_DATA__" 2>/dev/null; then
            print_success "Frontend appears to be a React/Next.js application"
        else
            print_info "Frontend is serving content (type unknown)"
        fi
        
        # Check for specific GeoPortugal indicators
        if echo "$frontend_response" | grep -qi "geoportugal\|portugal\|map" 2>/dev/null; then
            print_success "Frontend appears to be the GeoPortugal application"
        fi
    fi
    
    return 0
}

check_data_quality() {
    print_section "Data Quality & Coverage Tests"
    
    # Test data coverage - check if we have reasonable amounts of each type
    local districts_query="{\"query\": \"{ districts { name } }\"}"
    local districts_response
    districts_response=$(make_request "$GRAPHQL_ENDPOINT" "POST" "$districts_query" "Districts count")
    if [ $? -eq 0 ]; then
        local district_count
        district_count=$(extract_json_count "$districts_response" "name")
        print_info "Total districts: $district_count"
        
        if safe_compare "$district_count" "-ge" "18"; then
            print_success "District coverage looks good (Portugal has 18 districts)"
        else
            print_warning "District coverage may be incomplete ($district_count < 18)"
        fi
    fi
    
    # Check for essential cities
    local essential_cities=("Lisbon" "Porto" "Braga" "Coimbra" "Faro" "Aveiro")
    local missing_cities=0
    
    for city in "${essential_cities[@]}"; do
        print_info "Testing $city..."
        local city_test="{\"query\": \"{ citySearch(query: \\\"$city\\\") { name } }\"}"
        
        # Use curl directly for debugging
        local response=$(curl -s --max-time $TIMEOUT \
                       -X POST \
                       -H "Content-Type: application/json" \
                       -d "$city_test" \
                       "$GRAPHQL_ENDPOINT" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$response" ]; then
            local found_count
            found_count=$(extract_json_count "$response" "name")
            print_info "  Raw response: $(echo "$response" | head -c 150)..."
            print_info "  Extracted count: $found_count"
            
            if safe_compare "$found_count" "-gt" "0"; then
                print_success "✓ $city found"
            else
                print_warning "✗ $city missing (count: $found_count)"
                missing_cities=$((missing_cities + 1))
            fi
        else
            print_warning "✗ $city test failed - no response"
            missing_cities=$((missing_cities + 1))
        fi
    done
    
    if [ "$missing_cities" -eq 0 ]; then
        print_success "All essential cities found in database"
    else
        print_warning "$missing_cities essential cities are missing"
    fi
    
    return 0
}

check_performance() {
    print_section "Performance Tests"
    
    # Test response times under load (simple version)
    local total_time=0
    local test_queries=("Porto" "Lisboa" "Braga" "Coimbra" "Faro")
    
    for query in "${test_queries[@]}"; do
        local start_time=$(date +%s%N)
        local perf_query="{\"query\": \"{ citySearch(query: \\\"$query\\\") { name } }\"}"
        make_request "$GRAPHQL_ENDPOINT" "POST" "$perf_query" "Performance test: $query" >/dev/null 2>&1
        local end_time=$(date +%s%N)
        local query_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        total_time=$((total_time + query_time))
        
        if [ "$query_time" -lt 1000 ]; then
            print_success "$query search: ${query_time}ms (good)"
        elif [ "$query_time" -lt 3000 ]; then
            print_warning "$query search: ${query_time}ms (acceptable)"
        else
            print_error "$query search: ${query_time}ms (slow)"
        fi
    done
    
    local avg_time=$((total_time / ${#test_queries[@]}))
    print_info "Average query time: ${avg_time}ms"
    
    if [ "$avg_time" -lt 500 ]; then
        print_success "Overall performance is excellent"
    elif [ "$avg_time" -lt 1000 ]; then
        print_success "Overall performance is good"
    else
        print_warning "Performance may need optimization"
    fi
    
    return 0
}

check_docker_environment() {
    print_section "Docker Environment Status"
    
    # Check if Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        print_warning "Docker not available - skipping container checks"
        return 0
    fi
    
    # Check running containers
    print_info "Checking Docker containers..."
    
    local postgres_running=$(docker ps --filter "name=postgres" --format "{{.Names}}" 2>/dev/null | wc -l)
    local redis_running=$(docker ps --filter "name=redis" --format "{{.Names}}" 2>/dev/null | wc -l)
    
    if [ "$postgres_running" -gt 0 ]; then
        print_success "PostgreSQL container is running"
    else
        print_warning "PostgreSQL container not found (may be running natively)"
    fi
    
    if [ "$redis_running" -gt 0 ]; then
        print_success "Redis container is running"
    else
        print_warning "Redis container not found (may be running natively)"
    fi
    
    # Check Docker compose status if available
    if command -v docker-compose >/dev/null 2>&1; then
        if [ -f "docker-compose.yml" ]; then
            local compose_status=$(docker-compose ps 2>/dev/null | grep -c "Up" || echo "0")
            print_info "Docker Compose services running: $compose_status"
        fi
    fi
    
    return 0
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    local start_time=$(date +%s)
    local overall_status=0
    
    print_header "GeoPortugal System Health Check"
    print_info "Timestamp: $(date)"
    print_info "Testing endpoints:"
    print_info "  - Backend API: $REST_API_ENDPOINT"
    print_info "  - GraphQL: $GRAPHQL_ENDPOINT"
    print_info "  - Frontend: $FRONTEND_ENDPOINT"
    
    # Run all health checks
    check_docker_environment || overall_status=1
    check_backend_health || overall_status=1
    check_rest_api || overall_status=1
    check_graphql || overall_status=1
    check_data_quality || overall_status=1
    check_performance || overall_status=1
    check_frontend || overall_status=1
    
    # Final summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_header "Health Check Summary"
    
    if [ $overall_status -eq 0 ]; then
        print_success "All systems operational! ✨"
        print_info "Total check time: ${duration}s"
        echo -e "\n${GREEN}🎉 GeoPortugal is healthy and ready to serve users!${NC}"
    else
        print_error "Some issues detected during health check"
        print_info "Total check time: ${duration}s"
        echo -e "\n${RED}🚨 Please review the issues above and fix any problems${NC}"
    fi
    
    # Optional: Check for data quality report
    if [ -f "../backend/data/raw/data_quality_report.json" ]; then
        print_info "Data quality report available at: backend/data/raw/data_quality_report.json"
    fi
    
    exit $overall_status
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Check dependencies
if ! command -v curl >/dev/null 2>&1; then
    print_error "curl is required but not installed"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    print_warning "jq is not installed - JSON parsing will be limited"
fi

# Run main function
main "$@"
