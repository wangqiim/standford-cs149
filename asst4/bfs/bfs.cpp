#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
// #define VERBOSE 1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
#pragma omp parallel for schedule(dynamic, 100)
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {
                int index = __sync_fetch_and_add(&new_frontier->count, 1);
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}

int bottom_up_step(Graph g, int* distances, int frontier_distance) {
    int count = 0;
#pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < g->num_nodes; i++) {
        if (distances[i] != NOT_VISITED_MARKER) {
            // printf("skip vertex %d, count = %d\n", i, count);
            continue;
        }
        auto start_edge = incoming_begin(g, i);
        auto end_edge = incoming_end(g, i);
        for (auto iter_edge = start_edge; iter_edge < end_edge; iter_edge++) {
            if (distances[*iter_edge] == frontier_distance) {
                distances[i] = frontier_distance + 1;
                // printf("tag vertex %d\n", i);
#pragma omp atomic
                count++;
                break;
            }
        }
    }
    return count;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    sol->distances[ROOT_NODE_ID] = 0;

    int frontier_distance = 0;
    int count = 1;
    while (count < graph->num_nodes) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        int count_this_step = bottom_up_step(graph, sol->distances, frontier_distance++);
        count += count_this_step;
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("count=%-10d %.4f sec\n", count, end_time - start_time);
#endif
        if (count_this_step == 0) {
            break;
        }
    }
}

void hybrid_bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    int frontier_distance = distances[frontier->vertices[0]];
#pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < g->num_nodes; i++) {
        if (distances[i] != NOT_VISITED_MARKER) {
            // printf("skip vertex %d, count = %d\n", i, count);
            continue;
        }
        auto start_edge = incoming_begin(g, i);
        auto end_edge = incoming_end(g, i);
        for (auto iter_edge = start_edge; iter_edge < end_edge; iter_edge++) {
            if (distances[*iter_edge] == frontier_distance) {
                distances[i] = frontier_distance + 1;
                int index = __sync_fetch_and_add(&new_frontier->count, 1);
                new_frontier->vertices[index] = i;
                // printf("tag vertex %d\n", i);
                break;
            }
        }
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int visited_count;
    while (frontier->count != 0) {
        visited_count += frontier->count;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        if (frontier->count < (graph->num_nodes - visited_count)) {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        } else {
            hybrid_bottom_up_step(graph, frontier, new_frontier, sol->distances);
        }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
