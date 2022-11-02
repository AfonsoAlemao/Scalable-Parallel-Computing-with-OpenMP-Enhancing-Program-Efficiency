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
// frontier_count is != -1 for hybrid mode
bool top_down_step(
    Graph g,
    int* distances, int* outgoing_size, int numEdges, int dist_frontier, int *frontier_count, 
    int *search_max_in_frontier, int *search_min_in_frontier)
{
    // printf("%d\n", *search_max_in_frontier);
    int numNodes = g->num_nodes, max = -1, min = numNodes + 1;
    bool have_new_frontier = false;
    int new_frontier_count = 0;
    if (*search_max_in_frontier - *search_min_in_frontier > 8000) {
        #pragma omp parallel
        {
            int mycount = 0;
            int my_max = -1;
            int my_min = numNodes + 1;
            # pragma omp for schedule(dynamic, 8000) nowait
            for (int i = *search_min_in_frontier; i <= *search_max_in_frontier; i++) {
                
                if (distances[i] == dist_frontier) {
                    if (outgoing_size[i]) {
                        int start_edge = g->outgoing_starts[i];
                        int end_edge = (i == numNodes - 1)
                                        ? numEdges
                                        : g->outgoing_starts[i + 1];
                                        
                        // attempt to add all neighbors to the new frontier
                        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                            int outgoing = g->outgoing_edges[neighbor];

                            if (__sync_bool_compare_and_swap (&distances[outgoing], NOT_VISITED_MARKER, dist_frontier + 1)) {
                                // printf("adicionei %d\n", outgoing);
                                have_new_frontier = true;
                                if (*frontier_count != -1) {
                                    mycount++;
                                }
                                if (outgoing > my_max) {
                                    my_max = outgoing;
                                }
                                if (outgoing < my_min) {
                                    my_min = outgoing;
                                }
                            }
                        }
                    }
                }
                
            }
            if (*frontier_count != -1) {
                if (mycount > 0) {
                    #pragma omp atomic
                    new_frontier_count += mycount;
                }
            }
            if (my_max > max || my_min < min) {
                #pragma omp critical
                {
                    if (my_max > max) {
                        max = my_max;
                    }
                    if (my_min < min) {
                        min = my_min;
                    }
                }
            }
        }

    }
        
    else {
        for (int i = *search_min_in_frontier; i <= *search_max_in_frontier; i++) {
            if (distances[i] == dist_frontier) {
                if (outgoing_size[i]) {
                    int start_edge = g->outgoing_starts[i];
                    int end_edge = (i == numNodes - 1)
                                    ? numEdges
                                    : g->outgoing_starts[i + 1];
                                    
                    // attempt to add all neighbors to the new frontier
                    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                        int outgoing = g->outgoing_edges[neighbor];

                        if (distances[outgoing] == NOT_VISITED_MARKER) {
                            distances[outgoing] = dist_frontier + 1;
                            // printf("adicionei %d\n", outgoing);
                            have_new_frontier = true;
                            if (*frontier_count != -1) {
                                new_frontier_count++;
                            }
                            if (outgoing > max) {
                                max = outgoing;
                            }
                            if (outgoing < min) {
                                min = outgoing;
                            }
                        }
                    }
                }
            }
        }
    }

    *frontier_count = new_frontier_count;
    *search_max_in_frontier = max;
    *search_min_in_frontier = min;

    return have_new_frontier;
    
    

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
    int numEdges = graph->num_edges, dist_frontier = 0, flag = -1;
    bool have_new_frontier = true;
    int *outgoingsize;
    outgoingsize = (int*) malloc(sizeof(int) * numNodes);
    int search_max_in_frontier = 0, search_min_in_frontier = 0;

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        outgoingsize[i] = outgoing_size(graph, i);
    }

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    while (have_new_frontier) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        have_new_frontier = top_down_step(graph, sol->distances, outgoingsize, numEdges, dist_frontier, &flag, &search_max_in_frontier, &search_min_in_frontier);
        dist_frontier++;

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    }

    free(outgoingsize);
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
    Graph g, int* distances, int numEdges, int dist_frontier, int *frontier_count, int min, int max)
{
    /*for each vertex v in graph:
        if v has not been visited AND v shares an incoming edge with a vertex u on the frontier:
            add vertex v to frontier;*/

    int numNodes = g->num_nodes;
    // printf("dist_frontier = %d\n", dist_frontier);
    bool have_new_frontier = false;
    int new_frontier_count = 0;

    if (max - min > 8000) {
        # pragma omp parallel for schedule(dynamic, 8000)
        for (int i = min; i <= max; i++) {
            int mycount = 0;
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
                        // printf("Adicionei %d\n", incoming);
                        distances[i] = dist_frontier + 1; 
                        
                        have_new_frontier = true;

                        if (*frontier_count != -1) {
                            mycount++;
                        }
                        break;
                    }
                }
            }
            if (*frontier_count != -1) {
                if (mycount > 0) {
                    #pragma omp atomic
                    new_frontier_count += mycount;
                }
            }
        }
    }
    else {
        for (int i = min; i <= max; i++) {
            int mycount = 0;
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
                        // printf("Adicionei %d\n", incoming);
                        distances[i] = dist_frontier + 1; 
                        
                        have_new_frontier = true;

                        if (*frontier_count != -1) {
                            new_frontier_count++;
                        }
                        break;
                    }
                }
            }
        }
    }

    *frontier_count = new_frontier_count;

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
    int frontier_count = 1, distance_frontier = 0, flag = -1;
    int max = numNodes - 1, min = 1;

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
        // printf("max = %d, min = %d\n", max, min);
        have_new_frontier = bottom_up_step(graph, sol->distances, numEdges, distance_frontier, &flag, min, max);
        distance_frontier++;

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        if (have_new_frontier) {
            for (int i = min; i <= max; i++) {
                if (sol->distances[i] != NOT_VISITED_MARKER) {
                    min++;
                }
                else {
                    break;
                }
            }

            for (int i = max; i >= min; i--) {
                if (sol->distances[i] != NOT_VISITED_MARKER) {
                    max--;
                }
                else {
                    break;
                }
            }
        }

    }

}

void bfs_hybrid(Graph graph, solution* sol)
{
    // TODO STUDENTS:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    int numNodes = graph->num_nodes, frontier_count = 1;
    int numEdges = graph->num_edges, dist_frontier = 0; 
    int max = numNodes - 1, min = 1;
    bool have_new_frontier = true;
    int search_max_in_frontier = 0, search_min_in_frontier = 0;
    
    int *outgoingsize;
    outgoingsize = (int*) malloc(sizeof(int) * numNodes);

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        outgoingsize[i] = outgoing_size(graph, i);
        //outgoingstarts[i] = graph->outgoing_starts[i];
    }

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    while (have_new_frontier) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        // Top down complexity: number of frontier nodes
        // Bottom up complexity: number of nodes outside the bfs ~ number of nodes - number of frontier nodes
        if (frontier_count > numNodes / 8) {
            have_new_frontier = bottom_up_step(graph, sol->distances, numEdges, dist_frontier, &frontier_count, min, max);
            search_max_in_frontier = numNodes;
            search_min_in_frontier = 0;
        }
        else {
            have_new_frontier = top_down_step(graph, sol->distances, outgoingsize, numEdges, dist_frontier, &frontier_count, &search_max_in_frontier, &search_min_in_frontier);
        }
        dist_frontier++;

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    }

    free(outgoingsize);
}
