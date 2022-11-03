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

//#define VERBOSE
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
    int chunk_size = (frontier->count + 6400 - 1) / 6400;
    int dist_frontier = distances[frontier->vertices[0]];
    //printf("Frontier count %d\n", frontier->count);
    if (frontier->count > 1000) {
        int count = 0;
        # pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < frontier->count; i++) {
            int node = frontier->vertices[i];
            
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                int index = 0;

                if (__sync_bool_compare_and_swap (&distances[outgoing], NOT_VISITED_MARKER, dist_frontier + 1)) {                
                    # pragma omp atomic capture           
                    index = count++;

                    new_frontier->vertices[index] = outgoing;
                }
                
            }
        }
        new_frontier->count = count; 
    }
    else {
        for (int i = 0; i < frontier->count; i++) {
            int node = frontier->vertices[i];
            
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];


            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                int index = 0;

                if (distances[outgoing] == NOT_VISITED_MARKER) { 
                    distances[outgoing] = dist_frontier + 1;
                    index = new_frontier->count++;

                    new_frontier->vertices[index] = outgoing;
                }
            }
            
        }
    }
    // printf("Frontier count = %d\n", frontier->count);
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
    # pragma omp parallel for
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

    free(list1.vertices);
    free(list2.vertices);
}


// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    /*for each vertex v in graph:
        if v has not been visited AND v shares an incoming edge with a vertex u on the frontier:
            add vertex v to frontier;*/

    int dist_frontier = distances[frontier->vertices[0]];
    int chunk_size = (g->num_nodes + 6400 - 1) / 6400;
    // printf("dist_frontier = %d\n", dist_frontier);
    int count = 0;

    # pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < g->num_nodes; i++) {
        // printf("Tou no vertice %d\n", i);
        if (distances[i] == NOT_VISITED_MARKER) {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[i + 1];
            // printf("Numero vizinhos = %d\n", end_edge - start_edge);
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                int index = 0;
                // printf("Vizinho %d com distance = %d\n", incoming, distances[incoming]);
                if(distances[incoming] == dist_frontier) {
                    // printf("Adicionei ligacao %d-%d\n", i, incoming);
                    distances[i] = dist_frontier + 1;    
                    
                    # pragma omp atomic capture
                    index = count++;

                    new_frontier->vertices[index] = i;

                    break;
                }
            }
        }
    }

    new_frontier->count = count;

    // printf("\n\n");
}


void bfs_bottom_up(Graph graph, solution* sol)
{
    // TODO STUDENTS:
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
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for
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

        bottom_up_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers

        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(list1.vertices);
    free(list2.vertices);
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // TODO STUDENTS:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for
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
        // Top down complexity: number of frontier nodes
        // Bottom up complexity: number of nodes outside the bfs ~ number of nodes - number of frontier nodes
        if (frontier->count > graph->num_nodes / 8) {
            bottom_up_step(graph, frontier, new_frontier, sol->distances);
        }
        else {
            top_down_step(graph, frontier, new_frontier, sol->distances);
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

    free(list1.vertices);
    free(list2.vertices);
}
