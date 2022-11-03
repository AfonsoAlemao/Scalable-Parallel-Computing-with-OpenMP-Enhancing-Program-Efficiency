#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"
void bfs_top_down_hybrid(Graph graph, solution* sol);

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

// #define VERBOSE
void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

/* When we execute the # pragma omp for schedule(dynamic, chunk_size), it launches a “team” of OpenMP threads 
and partitions iterations among them. We choose dynamic schedule when the workload is not balanced across iterations. When a thread
finished one chunk, it is dynamically assigned another. The chunk size was chosen in order to guarantee enough 
granularity to balance the workload beetween threads and also to avoid that the overhead associated with the communication 
between threads and the launching of these be harmful to the performance of the program.*/

/* Take one step of "top-down" BFS for hybrid mode. 
  Performance is better if we iterate through the nodes rather than across the frontier, because
  in this way, for hybrid mode it is not necessary to explicitly represent the frontiers like in
  bottom up step. It is enough to represent the distances of each vertex to the root.
  frontier_count is only useful in hybrid mode (*frontier_count == -1 for the other modes). */ 
bool top_down_step_hybrid(
    Graph g,
    int* distances, int numEdges, int dist_frontier, int *frontier_count, 
    int *search_max_in_frontier, int *search_min_in_frontier)
{
    int numNodes = g->num_nodes, max = -1, min = numNodes + 1;
    bool have_new_frontier = false;
    int new_frontier_count = 0;

    /* Explained line 27. Hard coded value was chosen by testing. */
    int chunk_size = (numNodes + 6400 - 1) / 6400;

    /* To avoid that teams of OpenMP threads can be created and disbanded (or put in wait state) many times, 
        we want that a team created once are reused many times. Not active threads are put in wait state, 
        potentially reducing disbanding cost.  */
    #pragma omp parallel
    {
        int mycount = 0;
        int my_max = -1;
        int my_min = numNodes + 1;

        /* Removing implicit barrier. */
        # pragma omp for schedule(dynamic, chunk_size) nowait
        for (int i = *search_min_in_frontier; i <= *search_max_in_frontier; i++) {
            
            if (distances[i] == dist_frontier) {
                if (outgoing_size(g,i)) {
                    int start_edge = g->outgoing_starts[i];
                    int end_edge = (i == numNodes - 1)
                                    ? numEdges
                                    : g->outgoing_starts[i + 1];
                                    
                    /* Attempt to add all neighbors to the new frontier. */
                    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                        int outgoing = g->outgoing_edges[neighbor];

                        /* Must use atomic to avoid data races. */
                        if (__sync_bool_compare_and_swap (&distances[outgoing], NOT_VISITED_MARKER, dist_frontier + 1)) {
                            have_new_frontier = true;
                            /* Used to update the range of nodes that have not yet been visited */
                            if (outgoing > my_max) {
                                my_max = outgoing;
                            }
                            if (outgoing < my_min) {
                                my_min = outgoing;
                            }
                            /* Only for hybrid mode. */
                            if (*frontier_count != -1) {
                                mycount++;
                            }
                        }
                    }
                }
            }
            
        }
        /* Only for hybrid mode. */
        if (*frontier_count != -1) {
            if (mycount > 0) {
                /* Must use atomic to avoid data races. */
                #pragma omp atomic
                new_frontier_count += mycount;
            }
        }
        /* Updates the range of nodes that have not yet been visited */
        if (my_max > max || my_min < min) {
            /* Must use critical to avoid data races. */
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

    *frontier_count = new_frontier_count;
    *search_max_in_frontier = max;
    *search_min_in_frontier = min;

    return have_new_frontier;
}


/* Take one step of "top-down" BFS.  For each vertex on the frontier,
 follow all outgoing edges, and add all neighboring vertices to the
 new_frontier. Used for not dense graphs. */
bool top_down_step(
    Graph g,
    int **frontier,
    int **new_frontier,
    int *mycount_array,
    int dist_new_frontier,
    int *distances)
{
    bool have_frontier = false;
    int count[8];
    for (int i = 0; i < 8; i++) {
        count[i] = mycount_array[i];
    }

    /* To avoid that the teams of OpenMP threads can be created and disbanded (or put in wait state) many times, 
    we want that a team created once are reused many times. Not active threads are put in wait state, 
    potentially reducing disbanding cost.  */
    #pragma omp parallel 
    {        
        int tid = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        mycount_array[tid] = 0;

        for (int k = 0; k < 8; k++) {
            for (int i = tid; i < count[k] ; i += numThreads) {
                int node = frontier[k][i];
                int start_edge = g->outgoing_starts[node];
                int end_edge = (node == g->num_nodes - 1)
                                ? g->num_edges
                                : g->outgoing_starts[node + 1];

                /* Attempt to add all neighbors to the new frontier */
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int outgoing = g->outgoing_edges[neighbor];
                    int index = 0;

                    /* Must use atomic to avoid data races. */
                    if (distances[outgoing] == NOT_VISITED_MARKER) { 
                        distances[outgoing] = dist_new_frontier;
                        /* Must use critical to avoid data races. */
                        have_frontier = true;
                        new_frontier[tid][mycount_array[tid]++] = outgoing;
                        
                    }
                    
                }
            }
        }
        
    }

    return have_frontier;
   
}


/* Implements top-down BFS. Result of execution is that, for each node 
in the graph, the distance to the root is stored in sol.distances. */
void bfs_top_down(Graph graph, solution* sol) {

    int **frontier  = (int**) malloc(sizeof(int) * 8 * graph->num_nodes);
    int **new_frontier  = (int**) malloc(sizeof(int) * 8 * graph->num_nodes);
    int *mycount_array = (int*) calloc(sizeof(int), 8);
    bool have_frontier = true;

    for (int i = 0; i < 8; i++) {
        frontier[i]  = (int*) malloc(sizeof(int) * graph->num_nodes);
        new_frontier[i]  = (int*) malloc(sizeof(int) * graph->num_nodes);
    }


    /* Initialize all nodes to NOT_VISITED. The workload is balanced across iterations. */
    # pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    /* Setup frontier with the root node. */
    frontier[0][0] = ROOT_NODE_ID;
    mycount_array[0] = 1;
    // int sumcounts = 1;
    sol->distances[ROOT_NODE_ID] = 0;
    int dist_new_frontier = 1;

    while (have_frontier) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        have_frontier = top_down_step(graph, frontier, new_frontier, mycount_array, dist_new_frontier, sol->distances);

        dist_new_frontier++;

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        //sumcounts = 0;

        int **temp = frontier;
        frontier = new_frontier;
        new_frontier = temp;
    }
    
    for(int i = 0; i < 8; i++){
        free(frontier[i]);
        free(new_frontier[i]);
    }
    free(frontier);
    free(new_frontier);
    free(mycount_array);
}


/*for each vertex v in graph:
        if v has not been visited AND v shares an incoming edge with a vertex u on the frontier:
            add vertex v to frontier;*/
bool bottom_up_step(
    Graph g, int* distances, int numEdges, int dist_frontier, int *frontier_count, int min, int max)
{
    int numNodes = g->num_nodes;
    bool have_new_frontier = false;
    int new_frontier_count = 0;

    /* Explained line 27. Hard coded value was chosen by testing. */
    int chunk_size = (numNodes + 6400 - 1) / 6400;

    /* If the range of nodes that have not yet been visited is low, program's performance is improved
    if we execute our code sequentially, because of the overhead associated with the communication 
    between threads and its launching. */
    if (max - min > 8000) {
        /* To avoid that teams of OpenMP threads can be created and disbanded (or put in wait state) many times, 
        we want that a team created once are reused many times. Not active threads are put in wait state, 
        potentially reducing disbanding cost.  */
        # pragma omp parallel reduction(+:new_frontier_count)
        {
            int mycount = 0;
            # pragma omp for schedule(dynamic, chunk_size) nowait
            for (int i = min; i <= max; i++) {
                if (distances[i] == NOT_VISITED_MARKER) {
                    int start_edge = g->incoming_starts[i];
                    int end_edge = (i == numNodes - 1)
                                    ? numEdges
                                    : g->incoming_starts[i + 1];
                    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                        int incoming = g->incoming_edges[neighbor];

                        if (distances[incoming] == dist_frontier) {
                            distances[i] = dist_frontier + 1; 
                            
                            have_new_frontier = true;

                            /* Only for hybrid mode. */
                            if (*frontier_count != -1) {
                                mycount++;
                            }
                            break;
                        }
                    }
                }
            }
            /* Only for hybrid mode. */
            if (*frontier_count != -1) {
                if (mycount > 0) {
                    /* Must use atomic to avoid data races. */
                    new_frontier_count += mycount;
                }
            }
        }
    }
    else {
        for (int i = min; i <= max; i++) {
            int mycount = 0;
            if (distances[i] == NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == numNodes - 1)
                                ? numEdges
                                : g->incoming_starts[i + 1];
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int incoming = g->incoming_edges[neighbor];

                    if (distances[incoming] == dist_frontier) {
                        distances[i] = dist_frontier + 1; 
                        
                        have_new_frontier = true;

                        /* Only for hybrid mode. */
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
    // each step of the BFS process.frontier->count
void bfs_bottom_up(Graph graph, solution* sol)
{
    
    int numNodes = graph->num_nodes, numEdges = graph->num_edges;
    int frontier_count = 1, distance_frontier = 0, flag = -1;

    /* In this algorithm we always have to iterate over all nodes. To avoid that,
    boosting program's performance we limit the search by checking the range of 
    nodes that have not yet been visited. */
    int max = numNodes - 1, min = 1;

    /* Control variable to check if BFS computation has finished. */
    bool have_new_frontier = true;

    /* initialize all nodes to NOT_VISITED. The workload is balanced across iterations. */
    # pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;    
    }

    /* Setup frontier with the root node. */
    sol->distances[ROOT_NODE_ID] = 0;

    while (have_new_frontier) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        have_new_frontier = bottom_up_step(graph, sol->distances, numEdges, distance_frontier, &flag, min, max);
        distance_frontier++;

        /* Updates the range of nodes that have not yet been visited. */
        if (have_new_frontier) {
            while (sol->distances[min] != NOT_VISITED_MARKER && min <= max) {
                min++;
            }

            while (sol->distances[max] != NOT_VISITED_MARKER && min <= max) {
                max--;
            }
        }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier: %.4f sec\n", end_time - start_time);
#endif
    }

}


// TODO STUDENTS:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
void bfs_hybrid(Graph graph, solution* sol)
{
    int numNodes = graph->num_nodes, frontier_count = 1;
    int numEdges = graph->num_edges, dist_frontier = 0; 
    int max = numNodes - 1, min = 1;
    
    bool have_new_frontier = true;
    int search_max_in_frontier = 0, search_min_in_frontier = 0;
    
    /* Initialize all nodes to NOT_VISITED. The workload is balanced across iterations. */
    # pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    /* Setup frontier with the root node. */
    sol->distances[ROOT_NODE_ID] = 0;

    while (have_new_frontier) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        /* Top down complexity is given by number of frontier nodes. Bottom up complexity 
        is given by: number of nodes outside the bfs. So we can use frontier count as a metric 
        to decide whether it is more advantageous to perform bottom_up_step or top_down_step_dense
        for each iteration. To reconcile both steps we used top down step dense to avoid the need
        of frontier representation. */
        if (frontier_count > numNodes / 8) {
            have_new_frontier = bottom_up_step(graph, sol->distances, numEdges, dist_frontier, &frontier_count, min, max);
            search_max_in_frontier = numNodes;
            search_min_in_frontier = 0;
        }
        else {
            have_new_frontier = top_down_step_hybrid(graph, sol->distances, numEdges, dist_frontier, &frontier_count, &search_max_in_frontier, &search_min_in_frontier);
        }
        dist_frontier++;

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier_count, end_time - start_time);
#endif

    }

}