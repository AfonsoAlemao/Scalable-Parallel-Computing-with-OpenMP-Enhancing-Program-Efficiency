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
    int* distances, int* outgoing_size, int* outgoing_starts, int numEdges)
{
    int dist_frontier = distances[frontier->vertices[0]];
    int numNodes = g->num_nodes;
    int frontier_count = frontier->count;

    //printf("Frontier count %d\n", frontier_count);
    if (frontier_count > 1000) {
        # pragma omp parallel for schedule(dynamic, (frontier_count + 8000 - 1) / 8000)
        for (int i = 0; i < frontier_count; i++) {
            int outgoing_size_frontier = outgoing_size[frontier->vertices[i]];
            if (outgoing_size_frontier) {
                int myfrontiers[outgoing_size_frontier];
                int mycount = 0;
                int node = frontier->vertices[i];
                // printf("Tou no vertice %d\n", node);
                int start_edge = outgoing_starts[node];
                int end_edge = (node == numNodes - 1)
                                ? numEdges
                                : outgoing_starts[node + 1];
                                
                // attempt to add all neighbors to the new frontier
                // printf("Os meus vizinhos:\n");
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int outgoing = g->outgoing_edges[neighbor];
                    int index = 0;
                    // printf("%d\n", outgoing);

                    if (__sync_bool_compare_and_swap (&distances[outgoing], NOT_VISITED_MARKER, dist_frontier + 1)) {
                        myfrontiers[mycount++] = outgoing;
                        // printf("counters[%d] = %d\n", i, counters[i]);
                        // printf("Adicionei %d-%d\n", node, outgoing);
                    }
                }

                for (int j = 0; j < mycount; j++) {
                    // printf("Adicionei na new frontier %d-%d\n", frontier->vertices[i], frontiers[i][j]);
                    #pragma omp critical
                    {
                        new_frontier->vertices[new_frontier->count++] = myfrontiers[j];
                    }
                }
            }
        }

    }
    else {
        for (int i = 0; i < frontier_count; i++) {
            int outgoing_size_frontier = outgoing_size[frontier->vertices[i]];
            if (outgoing_size_frontier) {
                int myfrontiers[outgoing_size[frontier->vertices[i]]];
                int mycount = 0;
                int node = frontier->vertices[i];
                // printf("Tou no vertice %d\n", node);
                int start_edge = outgoing_starts[node];
                int end_edge = (node == numNodes - 1)
                                ? numEdges
                                : outgoing_starts[node + 1];
                                
                // attempt to add all neighbors to the new frontier
                // printf("Os meus vizinhos:\n");
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int outgoing = g->outgoing_edges[neighbor];
                    int index = 0;
                    // printf("%d\n", outgoing);

                    if (distances[outgoing] == NOT_VISITED_MARKER) {
                        distances[outgoing] = dist_frontier + 1;
                        myfrontiers[mycount++] = outgoing;
                        // printf("counters[%d] = %d\n", i, counters[i]);
                        // printf("Adicionei %d-%d\n", node, outgoing);
                    }
                }

                for (int j = 0; j < mycount; j++) {
                    // printf("Adicionei na new frontier %d-%d\n", frontier->vertices[i], frontiers[i][j]);
                    //#pragma omp critical
                    //{
                    new_frontier->vertices[new_frontier->count++] = myfrontiers[j];
                    //}
                }
            }
        }
    }
    

    /*if (numNodes > 10000) {
        # pragma omp parallel for schedule(dynamic, (numNodes + 8000 - 1) / 8000)
        for (int k = 0; k < numNodes; k++) {
            if (distances[k] == dist_frontier + 1) {
                int index = 0;
                #pragma omp critical 
                {
                    index = new_frontier->count++;
                }
                new_frontier->vertices[index] = k;
            }
        }
    }
    else {
        for (int k = 0; k < numNodes; k++) {
            if (distances[k] == dist_frontier + 1) {
                int index = new_frontier->count++;
                new_frontier->vertices[index] = k;
            }
        }
    }*/

    // printf("Frontier count = %d\n", frontier->count);
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    int numNodes = graph->num_nodes;
    int numEdges = graph->num_edges;
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, numNodes);
    vertex_set_init(&list2, numNodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    int *outgoingsize, *outgoingstarts;
    outgoingsize = (int*) malloc(sizeof(int) * numNodes);
    outgoingstarts = (int*) malloc(sizeof(int) * numNodes);

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        outgoingsize[i] = outgoing_size(graph, i);
        outgoingstarts[i] = graph->outgoing_starts[i];
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances, outgoingsize, outgoingstarts, numEdges);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers

        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(outgoingsize);
    free(outgoingstarts);
    free(list1.vertices);
    free(list2.vertices);
}

/**************************************************************************************
 * ProcuraBinaria ()
 *
 * Argumentos: a - tabela de items onde sera efetuada a procura
 *             l, r - limites da tabela
 *             x - elemento a pesquisar
 * 			   less, equal - funcoes com comparacoes a ser feitas para encontrar x em a
 * Retorna: inteiro com o indice onde x se encontra na tabela a,
 * 			ou -1 se nao for encontrado
 * Efeitos Secundarios: nenhum
 *
 * Descricao: caso encontre o elemento na tabela retorna indice, caso contrario -1
 * 			  O(lgN)
 **************************************************************************************/

/*int ProcuraBinaria (int a[], int l, int r, int x) {
    while (r >= l) {
		int m = (l + r) / 2;
		if (x == a[m]) return m;
		if (x < a[m]) r = m - 1;
		else l = m + 1;
	}
	return -1;
}*/

// return frontier_count
bool bottom_up_step(
    Graph g, int* distances, int numEdges, int dist_frontier)
{
    /*for each vertex v in graph:
        if v has not been visited AND v shares an incoming edge with a vertex u on the frontier:
            add vertex v to frontier;*/

    int numNodes = g->num_nodes;
    // printf("dist_frontier = %d\n", dist_frontier);
    bool have_new_frontier = false;

    # pragma omp for
    for (int i = 0; i < numNodes; i++) {
        // printf("Tou no vertice %d\n", i);
        if (distances[i] == NOT_VISITED_MARKER) {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == numNodes - 1)
                            ? numEdges
                            : g->incoming_starts[i + 1];
            // printf("Numero vizinhos = %d\n", end_edge - start_edge);
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];

                // printf("Vizinho %d com distance = %d\n", incoming, distances[incoming]);
                if (distances[incoming] == dist_frontier) {
                    // printf("Adicionei ligacao %d-%d\n", i, incoming);
                    distances[i] = dist_frontier + 1; 
                    
                    have_new_frontier = true;

                    break;
                }
            }
        }
    }

    return have_new_frontier;
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
    int numNodes = graph->num_nodes, numEdges = graph->num_edges;
    int frontier_count = 1, distance_frontier = 0;
    bool have_new_frontier = true;
    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;    
    }

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    while (have_new_frontier) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        have_new_frontier = bottom_up_step(graph, sol->distances, numEdges, distance_frontier);
        distance_frontier++;

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    }

}

void bfs_hybrid(Graph graph, solution* sol)
{
    // TODO STUDENTS:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    int numNodes = graph->num_nodes;
    int numEdges = graph->num_edges;

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, numNodes);
    vertex_set_init(&list2, numNodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    
    int *outgoingsize, *outgoingstarts;
    outgoingsize = (int*) malloc(sizeof(int) * numNodes);
    outgoingstarts = (int*) malloc(sizeof(int) * numNodes);

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        outgoingsize[i] = outgoing_size(graph, i);
        outgoingstarts[i] = graph->outgoing_starts[i];
    }

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
        //if (frontier->count * frontier->count > numNodes) {
        //    bottom_up_step(graph, frontier, new_frontier, sol->distances, numEdges);
        //}
        //else {
            top_down_step(graph, frontier, new_frontier, sol->distances, outgoingsize, outgoingstarts, numEdges);
        //}

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(outgoingsize);
    free(outgoingstarts);

    free(list1.vertices);
    free(list2.vertices);
}
