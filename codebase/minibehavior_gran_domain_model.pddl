;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  mini_behavior-domain.pddl   (revised July 2025)
;;
;;  • 15 primitives: left, right, forward, toggle, open, close,
;;    slice, cook, drop_in, pickup_0/1/2, drop_0/1/2.
;;  • Derived predicates compute reachability and cell-passability.
;;  • Static predicates reflect every per-object property from
;;    object_properties.json and object_action.json.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain mini_behavior)
  (:requirements 
    :strips 
    :typing 
    :conditional-effects 
    ; :quantified-preconditions 
    :derived-predicates
    :negative-preconditions 
    :equality 
    :disjunctive-preconditions 
    :existential-preconditions
    :universal-preconditions

    :action-costs
    
  )

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Types
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  (:types
    ;; Agent movement 
    location geo-info item-and-furniture - object

    direction dimension - geo-info
    normal-item furniture - item-and-furniture
    ;; Object and Furniture 
    agent apple backpack ball banana basket beef blender book bow bread cake calculator candy candle carton casserole chicken chip cookie date document dustpan egg fish floor folder fork gym_shoe hamburger hammer hardback highlighter jewelry juice kettle lemon lettuce marker necklace notebook oatmeal olive package pan pen pencil plate plywood pop pot_plant printer radish salad sandwich saw shoe soap sock soup spoon strawberry sugar tea_bag teapot toilet tomato vegetable_oil water window cutting-item cleaning-item - normal-item

    carving_knife knife - cutting-item
    broom rag scrub_brush towel - cleaning-item

    ashcan bed bin box bucket cabinet car chair countertop door electric_refrigerator wall shelf shower sink sofa stove table - furniture
  )

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Constants
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  (:constants
    north east south west - direction
    agent-01 - agent 
    bottom middle top - dimension
  )

  ; (:functions
  ;     (total-cost)- number
  ; )

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Predicates
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates


    ;; Object state 
    (is-cooked ?o - normal-item)

    (is-sliced ?o - normal-item)
    (is-not-dusted ?o - item-and-furniture)
    (is-not-stained ?o - item-and-furniture)
    (is-toggled ?o - item-and-furniture)
    (is-opened ?o - item-and-furniture)
    (is-soaked ?o - normal-item)
    (can-overlap ?o - furniture)
    (can-seebehind ?o - item-and-furniture)  
    (inhandofrobot ?ag - agent ?other - normal-item)

    ;; Object Default State (Derived)
    (atsamelocation ?o ?other - item-and-furniture)  ;; if two objects are at the same location, then they are in the same room
    (infovofrobot ?ag - agent ?other - item-and-furniture)
    (inreachofrobot ?ag - agent ?other - item-and-furniture)
    (insameroomasrobot ?ag - agent ?other - item-and-furniture)
    (inside ?obj - normal-item ?container - furniture ?loc - location ?dim - dimension)
    (nextto ?obj ?other - item-and-furniture)
    (onfloor ?obj - item-and-furniture) ;; if not in hand of robot then they are on the floor (derived predicate)
    (onTop ?obj ?other - item-and-furniture) ;; 1 not in hand of robot and 2 
    (under ?obj ?other - item-and-furniture) ;; similar to onTop, but the other way around
    (is-freezed ?o - normal-item) ;; if the object is in the cold source, then it is freezed
    ;; ------ End of Derived Predicates ------

    ;; Map topology (assert all in :init of problem)
    (move-dir ?from - location ?to - location ?d - direction)
    (succ ?l1 - geo-info ?l2 - geo-info) ;; successor relation for directions and dimensions
  

    ;; Object property need to init in the :init of problem
    (cookable ?o - normal-item)
    (freezable ?o - normal-item)
    (sliceable ?o - normal-item)
    (dustyable ?o - item-and-furniture)
    (stainable ?o - item-and-furniture)
    (toggleable ?o - item-and-furniture)
    (openable ?o - item-and-furniture)
    (soakable ?o - normal-item)
    (overlapable ?o - furniture) ;; if the object can overlap, then it is not overlapable anymore
    (containable ?o - furniture ?d - dimension) ;; it relates to inside predicate
    (seebehindable ?o - item-and-furniture) ;; 
    ;; pickable also means dropable and drop-inable, so we share the same predicate
    (pickable ?o - normal-item)
    

    ;; Object properties for specific tools
    (slicer ?o - cutting-item)
    (cleaningTool ?o - cleaning-item)
    (coldSource ?o - electric_refrigerator)
    (heatSource ?o - stove)
    (waterSource ?o - sink) ;; water source is sink

    ;; Dynamic state
    (facing ?ag - agent   ?d - direction)
    (at ?o  - item-and-furniture  ?l - location ?dim - dimension)

  )

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Derived definitions
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  ;; atsamelocation inreachofrobot inside nextto onfloor onTop under
  (:derived (atsamelocation ?o ?other - item-and-furniture)
    (and
      (exists (?sameloc - location ?odim ?otherdim - dimension) 
        (and
          (at ?o ?sameloc ?odim)
          (at ?other ?sameloc ?otherdim)
        )
      )
    )
  )

  (:derived (inreachofrobot ?ag - agent ?other - item-and-furniture)
    (and 
      (exists (?agloc ?otherloc - location ?dim - dimension ?dir - direction)
        (and
          (at ?other ?otherloc ?dim) ;; the object is at the same location and dimension
          (at ?ag ?agloc bottom) ;; the agent is at the same location but bottom layer
          (facing ?ag ?dir) ;; the agent is facing a direction
          (move-dir ?agloc ?otherloc ?dir) ;; the agent can reach the object in that direction
          ; (not (exists (?cnt - furniture)
          ;   (and
          ;     (not (= ?cnt ?other)) ;; the container is not the object itself
          ;     (containable ?cnt ?dim) ;; the container can contain at that dimension
          ;     (at ?cnt ?otherloc ?dim) ;; the container is at the same location and dimension as the object
          ;     (openable ?cnt) ;; the container is openable
          ;     (not (is-opened ?cnt)) ;; the container is not opened
          ;   )
          ;  ))
        )
      )
    )
  )

  (:derived (inside ?obj - normal-item ?container - furniture ?loc - location ?dim - dimension)
    (and
      (containable ?container ?dim) ;; the container is containable at the specified dimension
      (at ?obj  ?loc ?dim)
      (at ?container ?loc ?dim) ;; the container is at the same location and dimension
    )
  )

  (:derived (nextto ?obj ?other - item-and-furniture)
    (and
      (exists (?objloc ?otherloc - location ?odim ?otherdim - dimension ?dir - direction)
        (and
          (at ?obj ?objloc ?odim) ;; the object is at the same location and dimension
          (at ?other ?otherloc ?otherdim) ;; the other object is at the same location and dimension
          (move-dir ?objloc ?otherloc ?dir) ;; there is a path from obj to other in the direction dir
          (not (= ?obj ?other)) ;; the two objects are not the same
          (not (at ?other ?objloc bottom))
          (not (at ?other ?objloc middle))
          (not (at ?other ?objloc top)) ;; the other object is not at the same location and dimension as the object
        )
      )
    )
  )

  (:derived (onfloor ?obj - item-and-furniture)
    (not (exists (?ag - agent)
        (and
          (inhandofrobot ?ag ?obj) ;; the object is in hand of robot
        )
      )
    )
  )

  (:derived (onTop ?obj ?other - item-and-furniture)
    (or 
      (exists (?loc - location ?otherdim - dimension)
        (and
          (at ?obj ?loc top) ;; the object is at the top layer at the same location
          (at ?other ?loc ?otherdim) ;; the other object is at the middle layer at the same location
          (not (= ?otherdim top))
        )
      )
      (exists (?loc - location)
        (and
          (at ?obj ?loc middle) ;; the object is at the middle layer at the same location
          (at ?other ?loc bottom) ;; the other object is at the bottom layer at the same location
        )
      )
    )
  )

  (:derived (under ?obj ?other - item-and-furniture)
    (or
      (exists (?loc - location ?otherdim - dimension)
        (and
          (at ?obj ?loc bottom) ;; the object is at the bottom layer at the same location
          (at ?other ?loc ?otherdim) ;; the other object is at the middle layer at the same location
          (not (= ?otherdim bottom))
        )
      )
      (exists (?loc - location)
        (and
          (at ?obj ?loc middle) ;; the object is at the middle layer at the same location
          (at ?other ?loc top) ;; the other object is at the top layer at the same location
        )
      )
    )
  )

  (:derived (is-freezed ?o - normal-item)
    (and
      (freezable ?o) ;; the object is freezable
      (exists (?coldobj - electric_refrigerator ?sameloc - location ?odim - dimension) 
        (and
          (coldSource ?coldobj) ;; cold source
          (at ?o ?sameloc ?odim)
          (containable ?coldobj ?odim) ;; the cold source can contain the object at the same dimension
          (at ?coldobj ?sameloc ?odim) ;; the cold source is at the same location and dimension
      
        )
      )
    )
  )



  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Primitive actions
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; ── Movement ───────────────────────────────────────────────────────────────
  (:action turn-left
    :parameters (?ag - agent ?loc - location ?cur_dir - direction ?new_dir - direction)
    :precondition (and
      (at ?ag ?loc bottom)
      (succ ?new_dir ?cur_dir) ;; left turn so new_dir is on the front of cur_dir
      (facing ?ag ?cur_dir) ;; agent is facing the current direction
    )
    :effect (and 
      (not (facing ?ag ?cur_dir))
      (facing ?ag ?new_dir) ;; agent is now facing the new direction
      (increase (total-cost) 1)
    )
  )

  (:action turn-right
    :parameters (?ag - agent ?loc - location ?cur_dir - direction ?new_dir - direction)
    :precondition (and
      (at ?ag ?loc bottom)
      (succ ?cur_dir ?new_dir) ;; right turn so new_dir is on the back of cur_dir
      (facing ?ag ?cur_dir) ;; agent is facing the current direction
    )
    :effect (and 
      (not (facing ?ag ?cur_dir))
      (facing ?ag ?new_dir) ;; agent is now facing the new direction
      (increase (total-cost) 1)
    )
  )

  (:action forward
    :parameters (?ag - agent ?from - location ?to - location ?dir - direction)
    :precondition (and
      (at ?ag ?from bottom)
      (facing ?ag ?dir)
      (move-dir ?from ?to ?dir) ;; there is a path from from to to in the direction dir
      ;; there is no object in the ?to location such that (not (can-overlap ?thatobj))
      (not (exists (?someobj - item-and-furniture ?somedim - dimension)
        (and 
          (at ?someobj ?to ?somedim)
          (not (can-overlap ?someobj))
     )
      ))
    )
    :effect (and
      (not (at ?ag ?from bottom))
      (at ?ag ?to bottom)     
      (increase (total-cost) 1) 
    )
  )

  ;; ── Obj Manipulation Actions ───────────────────────────────────────────────────────────────

  (:action close
    :parameters (?ag - agent ?obj - item-and-furniture ?ag-loc ?obj-loc ?dir - direction)
    :precondition (and
      (openable ?obj)
      (is-opened ?obj)
      (inreachofrobot ?ag ?obj)
    )
    :effect (and
      (not (is-opened ?obj))
      (when (overlapable ?obj) (not (can-overlap ?obj))) ;; if the object is overlapable, then it is not can-overlap anymore
      (when (seebehindable ?obj) (not (can-seebehind ?obj))) ;; if the object is seebehindable, then it is not can-seebehind anymore
      (increase (total-cost) 10)
    )
  )

  ; (:action cook
  ;   :parameters (?ag - agent ?food - normal-item ?p - pan ?heat-source - furniture)
  ;   :precondition (and
  ;     ;; - food is cookable
  ;     ;; - agent is carrying a cooking tool
  ;     ;; - agent is infront of a heat source
  ;     ;; - the heat source is toggled on
  ;     (cookable ?food) ; food is cookable
  ;     (inhandofrobot ?ag ?p) ; agent is holding a pan
  ;     (inreachofrobot ?ag ?heat-source) ; agent is in reach of the heat source
  ;     (inreachofrobot ?ag ?food) ; agent is in reach of the food, in fact, food should be inside of the heat-source 
  ;     (exists (?dim - dimension ?loc - location) ;; there is a location and dimension where the food is inside the heat source
  ;       (and
  ;         (inside ?food ?heat-source ?loc ?dim)
  ;       )
  ;     )
  ;     (is-toggled ?heat-source) ; heat source is toggled on
  ;     (heatSource ?heat-source) ; heat source is a heat source (e.g., stove)
  ;   )
  ;   :effect (and 
  ;     (is-cooked ?food) ; food is now cooked
  ;     (increase (total-cost) 10)
  ;   )
  ; )


  ;; ── Drop (by layer) ───────────────────────────────────────────────────────
  (:action drop_0  ;; 0 means bottom layer
    :parameters (?ag - agent ?o - normal-item ?agloc ?targetloc - location ?cur_dir - direction)
    :precondition (and
      ;; - agent is holding the object
      (inhandofrobot ?ag ?o)
      ;; - agent is at the location
      (at ?ag ?agloc bottom)
      ;; - agent is facing the direction
      (facing ?ag ?cur_dir)
      ;; - the target location is adjacent to the agent's location in the direction
      (move-dir ?agloc ?targetloc ?cur_dir)
      ;; there is no object also in the target location bottom layer
      (not (exists (?someobj - item-and-furniture) 
        (and 
          (at ?someobj ?targetloc bottom)
        )
      ))
    )
    :effect (and
      ;; object is no longer in hand of robot
      (not (inhandofrobot ?ag ?o))
      ;; object is now at the target location
      (at ?o ?targetloc bottom)

      ;; AUTO CASE: handle is-not-dusted predicate
      (forall (?dustobj - item-and-furniture) ;; in this case ?o is cleanobj
        (when (and
          (dustyable ?dustobj)
          (cleaningTool ?o)
          (exists (?otherobjdim - dimension)
            (at ?dustobj ?targetloc ?otherobjdim) ;; the dustable object is at the target location
          )
          (not (is-not-dusted ?dustobj)) ;; the dustable object is not dusted yet
        )
          (is-not-dusted ?dustobj) ;; the dustable object is not dusted anymore
        )
      )

      (when (and  ;; in this case ?o is dustobj
        (dustyable ?o) ;; the object is dustable
        (exists (?cleanobj - normal-item)
          (and
            (cleaningTool ?cleanobj)
            (exists (?otherobjdim - dimension)
              (at ?cleanobj ?targetloc ?otherobjdim) ;; the cleaning tool is at the target location
            )
          )
        )
        (not (is-not-dusted ?o)) ;; the object is not dusted yet
      )
        (is-not-dusted ?o) ;; the object is not dusted anymore
      )

      ;; AUTO CASE: handle is-not-stained predicate
      (forall (?stainobj - normal-item) ;; in this case ?o is cleanobj
        (when (and
          (stainable ?stainobj)       ;; stainable object
          (cleaningTool ?o)           ;; cleaning tool
          (soakable ?o)               ;; cleaning tool is soakable
          (is-soaked ?o)              ;; cleaning tool is soaked
          (exists (?otherobjdim - dimension)
            (at ?stainobj ?targetloc ?otherobjdim) ;; the stainable object is at the target location
          )
          (not (is-not-stained ?stainobj)) ;; the stainable object is not stained yet
        )
          (is-not-stained ?stainobj) ;; the stainable object is not stained anymore
        )
      )

      (when (and  ;; in this case ?o is stainobj
        (stainable ?o)                  ;; the object is stainable

        (exists (?cleanobj - normal-item)
          (and
            (cleaningTool ?cleanobj)   ;; cleaning tool
            (soakable ?cleanobj)       ;; cleaning tool is soakable
            (is-soaked ?cleanobj)      ;; cleaning tool is soaked
            (exists (?otherobjdim - dimension)
              (at ?cleanobj ?targetloc ?otherobjdim) ;; the cleaning tool is at the target location
            )
          )
        )
        (not (is-not-stained ?o))       ;; the object is not stained yet
      )
        (is-not-stained ?o)             ;; the object is not stained anymore
      )
      (increase (total-cost) 10)
    )
  )

  (:action drop_1 ;; 1 means middle layer
    :parameters (?ag - agent ?o - normal-item ?agloc ?targetloc - location ?cur_dir - direction)
    :precondition (and
      ;; - agent is holding the object
      (inhandofrobot ?ag ?o)
      ;; - agent is at the location
      (at ?ag ?agloc bottom)
      ;; - agent is facing the direction
      (facing ?ag ?cur_dir)
      ;; - the target location is adjacent to the agent's location in the direction
      (move-dir ?agloc ?targetloc ?cur_dir)
      ;; there is no object also in the target location middle layer
      (not (exists (?someobj - item-and-furniture) 
        (and 
          (at ?someobj ?targetloc middle)
        )
      ))
      ;; there exists an object at the target location bottom layer
      (exists (?someobj - item-and-furniture)
        (and
          (at ?someobj ?targetloc bottom)
        )
      )
    )
    :effect (and
      ;; object is no longer in hand of robot
      (not (inhandofrobot ?ag ?o))
      ;; object is now at the target location
      (at ?o ?targetloc middle)

      ;; AUTO CASE: handle is-not-dusted predicate
      (forall (?dustobj - item-and-furniture) ;; in this case ?o is cleanobj
        (when (and
          (dustyable ?dustobj)
          (cleaningTool ?o)
          (exists (?otherobjdim - dimension)
            (at ?dustobj ?targetloc ?otherobjdim) ;; the dustable object is at the target location
          )
          (not (is-not-dusted ?dustobj)) ;; the dustable object is not dusted yet
        )
          (is-not-dusted ?dustobj) ;; the dustable object is not dusted anymore
        )
      )

      (when (and  ;; in this case ?o is dustobj
        (dustyable ?o) ;; the object is dustable
        (exists (?cleanobj - normal-item)
          (and
            (cleaningTool ?cleanobj)
            (exists (?otherobjdim - dimension)
              (at ?cleanobj ?targetloc ?otherobjdim) ;; the cleaning tool is at the target location
            )
          )
        )
        (not (is-not-dusted ?o)) ;; the object is not dusted yet
      )
        (is-not-dusted ?o) ;; the object is not dusted anymore
      )

      ;; AUTO CASE: handle is-not-stained predicate
      (forall (?stainobj - normal-item) ;; in this case ?o is cleanobj
        (when (and
          (stainable ?stainobj)       ;; stainable object
          (cleaningTool ?o)           ;; cleaning tool
          (soakable ?o)               ;; cleaning tool is soakable
          (is-soaked ?o)              ;; cleaning tool is soaked
          (exists (?otherobjdim - dimension)
            (at ?stainobj ?targetloc ?otherobjdim) ;; the stainable object is at the target location
          )
          (not (is-not-stained ?stainobj)) ;; the stainable object is not stained yet
        )
          (is-not-stained ?stainobj) ;; the stainable object is not stained anymore
        )
      )

      (when (and  ;; in this case ?o is stainobj
        (stainable ?o)                  ;; the object is stainable

        (exists (?cleanobj - normal-item)
          (and
            (cleaningTool ?cleanobj)   ;; cleaning tool
            (soakable ?cleanobj)       ;; cleaning tool is soakable
            (is-soaked ?cleanobj)      ;; cleaning tool is soaked
            (exists (?otherobjdim - dimension)
              (at ?cleanobj ?targetloc ?otherobjdim) ;; the cleaning tool is at the target location
            )
          )
        )
        (not (is-not-stained ?o))       ;; the object is not stained yet
      )
        (is-not-stained ?o)             ;; the object is not stained anymore
      )
      (increase (total-cost) 10)
    )
  )

  (:action drop_2 ;; 2 means top layer
    :parameters (?ag - agent ?o - normal-item ?agloc ?targetloc - location ?cur_dir - direction)
    :precondition (and
      ;; - agent is holding the object
      (inhandofrobot ?ag ?o)
      ;; - agent is at the location
      (at ?ag ?agloc bottom)
      ;; - agent is facing the direction
      (facing ?ag ?cur_dir)
      ;; - the target location is adjacent to the agent's location in the direction
      (move-dir ?agloc ?targetloc ?cur_dir)
      ;; there is no object also in the target location top layer
      (not (exists (?someobj - item-and-furniture) 
        (and 
          (at ?someobj ?targetloc top)
        )
      ))
      ;; there exists an object at the target location middle layer
      (exists (?someobj - item-and-furniture)
        (and
          (at ?someobj ?targetloc middle)
        )
      )
    )
    :effect (and
      ;; object is no longer in hand of robot
      (not (inhandofrobot ?ag ?o))
      ;; object is now at the target location
      (at ?o ?targetloc top)

      ;; AUTO CASE: handle is-not-dusted predicate
      (forall (?dustobj - item-and-furniture) ;; in this case ?o is cleanobj
        (when (and
          (dustyable ?dustobj)
          (cleaningTool ?o)
          (exists (?otherobjdim - dimension)
            (at ?dustobj ?targetloc ?otherobjdim) ;; the dustable object is at the target location
          )
          (not (is-not-dusted ?dustobj)) ;; the dustable object is not dusted yet
        )
          (is-not-dusted ?dustobj) ;; the dustable object is not dusted anymore
        )
      )

      (when (and  ;; in this case ?o is dustobj
        (dustyable ?o) ;; the object is dustable
        (exists (?cleanobj - normal-item)
          (and
            (cleaningTool ?cleanobj)
            (exists (?otherobjdim - dimension)
              (at ?cleanobj ?targetloc ?otherobjdim) ;; the cleaning tool is at the target location
            )
          )
        )
        (not (is-not-dusted ?o)) ;; the object is not dusted yet
      )
        (is-not-dusted ?o) ;; the object is not dusted anymore
      )

      ;; AUTO CASE: handle is-not-stained predicate
      (forall (?stainobj - normal-item) ;; in this case ?o is cleanobj
        (when (and
          (stainable ?stainobj)       ;; stainable object
          (cleaningTool ?o)           ;; cleaning tool
          (soakable ?o)               ;; cleaning tool is soakable
          (is-soaked ?o)              ;; cleaning tool is soaked
          (exists (?otherobjdim - dimension)
            (at ?stainobj ?targetloc ?otherobjdim) ;; the stainable object is at the target location
          )
          (not (is-not-stained ?stainobj)) ;; the stainable object is not stained yet
        )
          (is-not-stained ?stainobj) ;; the stainable object is not stained anymore
        )
      )

      (when (and  ;; in this case ?o is stainobj
        (stainable ?o)                  ;; the object is stainable

        (exists (?cleanobj - normal-item)
          (and
            (cleaningTool ?cleanobj)   ;; cleaning tool
            (soakable ?cleanobj)       ;; cleaning tool is soakable
            (is-soaked ?cleanobj)      ;; cleaning tool is soaked
            (exists (?otherobjdim - dimension)
              (at ?cleanobj ?targetloc ?otherobjdim) ;; the cleaning tool is at the target location
            )
          )
        )
        (not (is-not-stained ?o))       ;; the object is not stained yet
      )
        (is-not-stained ?o)             ;; the object is not stained anymore
      )
      (increase (total-cost) 10)
    )
  )

  ;; ── Drop-in (inside an open container) ────────────────────────────────────
  (:action drop_in
    :parameters (?ag - agent ?o - normal-item ?cnt - furniture ?agentloc ?l - location ?d - dimension ?agent-dir - direction)
    :precondition (and
      (inhandofrobot ?ag ?o) ;; agent is holding the object
      (inreachofrobot ?ag ?cnt) ;; agent is in reach of the container
      (move-dir ?agentloc ?l ?agent-dir) ;; agent is facing the direction to the location
      (facing ?ag ?agent-dir)
      (at ?ag ?agentloc bottom) ;; agent is at the location at the bottom layer
      (exists (?cnt-dim - dimension) ;; there is a dimension of the container
        (and
          (at ?cnt ?l ?cnt-dim) ;; container is at the location and dimension
        )
      )
      (containable ?cnt ?d) ;; container is containable at the specified dimension
      ;; if openable, then it must be opened
      (imply (openable ?cnt) (is-opened ?cnt)) ;; if the container is openable, then it must be opened
      ;; if ?d is not bottom, then there exists an object ?otherobj at the location ?l and the ?otherdim dimension where succ ?otherdim ?d hold 
      ;; i.e., the object should stack on the top of the container
      (imply (not (= ?d bottom))
        (exists (?otherobj - normal-item ?otherdim - dimension)
          (and
            (at ?otherobj ?l ?otherdim)
            (succ ?otherdim ?d) ;; ?d is on the top of ?otherdim
          )
      ))
      ; there is no other item occupying the same location and dimension
      (not (exists (?someobj - normal-item)
            (and 
              (at ?someobj ?l ?d) ;; there is an object at the location and dimension
            )
          )
      )

    )
    :effect (and
      (not (inhandofrobot ?ag ?o))
      (at ?o ?l ?d) ;; object is now at the location and dimension

      ;; AUTO CASE: handle is-not-dusted predicate
      (forall (?dustobj - item-and-furniture) ;; in this case ?o is cleanobj
        (when (and
          (dustyable ?dustobj)
          (cleaningTool ?o)
          (exists (?otherobjdim - dimension)
            (at ?dustobj ?l ?otherobjdim) ;; the dustable object is at the target location
          )
          (not (is-not-dusted ?dustobj)) ;; the dustable object is not dusted yet
        )
          (is-not-dusted ?dustobj) ;; the dustable object is not dusted anymore
        )
      )

      (when (and  ;; in this case ?o is dustobj
        (dustyable ?o) ;; the object is dustable
        (exists (?cleanobj - normal-item)
          (and
            (cleaningTool ?cleanobj)
            (exists (?otherobjdim - dimension)
              (at ?cleanobj ?l ?otherobjdim) ;; the cleaning tool is at the target location
            )
          )
        )
        (not (is-not-dusted ?o)) ;; the object is not dusted yet
      )
        (is-not-dusted ?o) ;; the object is not dusted anymore
      )

      ;; AUTO CASE: handle is-not-stained predicate
      (forall (?stainobj - normal-item) ;; in this case ?o is cleanobj
        (when (and
          (stainable ?stainobj)       ;; stainable object
          (cleaningTool ?o)           ;; cleaning tool
          (soakable ?o)               ;; cleaning tool is soakable
          (is-soaked ?o)              ;; cleaning tool is soaked
          (exists (?otherobjdim - dimension)
            (at ?stainobj ?l ?otherobjdim) ;; the stainable object is at the target location
          )
          (not (is-not-stained ?stainobj)) ;; the stainable object is not stained yet
        )
          (is-not-stained ?stainobj) ;; the stainable object is not stained anymore
        )
      )

      (when (and  ;; in this case ?o is stainobj
        (stainable ?o)                  ;; the object is stainable

        (exists (?cleanobj - normal-item)
          (and
            (cleaningTool ?cleanobj)   ;; cleaning tool
            (soakable ?cleanobj)       ;; cleaning tool is soakable
            (is-soaked ?cleanobj)      ;; cleaning tool is soaked
            (exists (?otherobjdim - dimension)
              (at ?cleanobj ?l ?otherobjdim) ;; the cleaning tool is at the target location
            )
          )
        )
        (not (is-not-stained ?o))       ;; the object is not stained yet
      )
        (is-not-stained ?o)             ;; the object is not stained anymore
      )

      ;; AUTO CASE 3: handle soaked predicate
      (when (and
          (soakable ?o) ;; the object is soakable
          (not (is-soaked ?o)) ;; the object is not soaked yet
          (exists (?waterobj - item-and-furniture ?waterdim - dimension) 
            (and
              (waterSource ?waterobj) ;; water source
              (at ?waterobj ?l ?waterdim)
              (imply (toggleable ?waterobj) (is-toggled ?waterobj)) ;; if the water source is toggleable, then it must be toggled on
            )
          )
        )
          (is-soaked ?o) ;; the soakable object is now soaked
      )
      (increase (total-cost) 10)

    )
  )

  ;; ── Open  containers & doors ────────────────────────────────────────
  (:action open
    :parameters (?ag - agent ?c - item-and-furniture)
    :precondition (and
      (inreachofrobot ?ag ?c)
      (openable ?c)
      (not (is-opened ?c)) ;; the container or door is not opened yet
    )
    :effect (and
      (is-opened ?c) ;; the container or door is now opened
      (when (overlapable ?c) (can-overlap ?c)) ;; if the object is overlapable, then it is can-overlap again
      (when (seebehindable ?c) (can-seebehind ?c)) ;; if the object is seebehindable, then it is can-seebehind again
      (increase (total-cost) 10)
    )
  )


    ;; ── Pickup (by layer) ─────────────────────────────────────────────────────
  (:action pickup_0_container ;; 0 means bottom layer
    :parameters (?ag - agent ?o - normal-item ?l - location ?c - furniture)
    :precondition (and
      (pickable ?o) ;; object is pickable
      (inreachofrobot ?ag ?o) ;; agent is in reach of the object
      (not (inhandofrobot ?ag ?o)) ;; agent is not already holding the object
      (at ?o ?l bottom) ;; object is at the location and bottom layer
      (at ?c ?l bottom) ;; container is at the same location and bottom layer
      (containable ?c bottom) ;; container is containable at the bottom layer
      (imply (openable ?c) (is-opened ?c)) ;; if the container is openable, then it must be opened
      ;; no other objects currently inhandofrobot
      (not (exists (?otherobj - normal-item)
        (and 
          (inhandofrobot ?ag ?otherobj) ;; agent is holding another object
        )
      ))
    )
    :effect (and
      (not (at ?o ?l bottom)) ;; object is no longer at the location and bottom layer
      (inhandofrobot ?ag ?o) ;; agent is now holding the object
      ;; forall objects that at the middle layer and at the same location, shift them to the bottom layer
      (forall (?otherobj - normal-item)
        (when (and
          (at ?otherobj ?l middle) ;; other object is at the middle layer at the same location
          (not (= ?otherobj ?o)) ;; other object is not the same as the object being picked up
          ;; meanwhile also inside the same container
          (containable ?c middle) ;; container is containable at the middle layer
        )
          (and
            (at ?otherobj ?l bottom) ;; shift the other object to bottom layer
            (not (at ?otherobj ?l middle)) ;; other object is no longer at the middle layer
          )
        ) 
      )
      ;; forall objects that at the top layer and at the same location, shift them to the middle layer
      (forall (?otherobj - normal-item)
        (when (and
          (at ?otherobj ?l top) ;; other object is at the top layer at the same location
          (not (= ?otherobj ?o)) ;; other object is not the same as the object being picked up
          ;; meanwhile also inside the same container
          (containable ?c top) ;; container is containable at the top layer
        )
          (and
            (at ?otherobj ?l middle) ;; shift the other object to middle layer
            (not (at ?otherobj ?l top)) ;; other object is no longer at the top layer
          )
        ) 
      )
      (increase (total-cost) 10)

    )
  )

  (:action pickup_1_container ;; 1 means middle layer
    :parameters (?ag - agent ?o - normal-item ?l - location ?c - furniture)
    :precondition (and
      (pickable ?o) ;; object is pickable
      (inreachofrobot ?ag ?o) ;; agent is in reach of the object
      (not (inhandofrobot ?ag ?o)) ;; agent is not already holding the object
      (at ?o ?l middle) ;; object is at the location and middle layer
      (at ?c ?l middle) ;; container is at the same location and middle layer
      (containable ?c middle) ;; container is containable at the middle layer
      (imply (openable ?c) (is-opened ?c)) ;; if the container is openable, then it must be opened
      ;; no other objects currently inhandofrobot
      (not (exists (?otherobj - normal-item)
        (and 
          (inhandofrobot ?ag ?otherobj) ;; agent is holding another object
        )
      ))
    )
    :effect (and
      (not (at ?o ?l middle)) ;; object is no longer at the location and middle layer
      (inhandofrobot ?ag ?o) ;; agent is now holding the object
      ;; forall objects that at the top layer and at the same location, shift them to middle layer
      (forall (?otherobj - normal-item)
        (when (and
          (at ?otherobj ?l top) ;; other object is at the top layer at the same location
          (not (= ?otherobj ?o)) ;; other object is not the same as the object being picked up
          ;; meanwhile also inside the same container
          (containable ?c top) ;; container is containable at the top layer
        )
          (and
            (at ?otherobj ?l middle) ;; shift the other object to middle layer
            (not (at ?otherobj ?l top)) ;; other object is no longer at the top layer
          )
        ) 
      )
      (increase (total-cost) 10)

    )
  
  )

    (:action pickup_0 ;; 0 means bottom layer
    :parameters (?ag - agent ?o - normal-item ?l - location)
    :precondition (and
      (pickable ?o) ;; object is pickable
      (inreachofrobot ?ag ?o) ;; agent is in reach of the object
      (not (inhandofrobot ?ag ?o)) ;; agent is not already holding the object
      (at ?o ?l bottom) ;; object is at the location and bottom layer
      (not (exists (?c - furniture)
        (and
          (at ?c ?l bottom) ;; container is at the same location and bottom layer
          (containable ?c bottom) ;; container is containable at the bottom layer
        )
      ))
      ;; no other objects currently inhandofrobot
      (not (exists (?otherobj - normal-item)
        (and 
          (inhandofrobot ?ag ?otherobj) ;; agent is holding another object
        )
      ))
    )
    :effect (and
      (not (at ?o ?l bottom)) ;; object is no longer at the location and bottom layer
      (inhandofrobot ?ag ?o) ;; agent is now holding the object
      ;; forall objects that at the middle layer and at the same location, shift them to the bottom layer
      (forall (?otherobj - normal-item)
        (when (and
          (at ?otherobj ?l middle) ;; other object is at the middle layer at the same location
          (not (= ?otherobj ?o)) ;; other object is not the same as the object being picked up
          ;; meanwhile also inside the same container
          (not (exists (?container - furniture)
            (and
              (at ?container ?l middle) ;; container is at the same location and middle layer
              (containable ?container middle) ;; container is containable at the middle layer
            )
          ))
        )
          (and
            (at ?otherobj ?l bottom) ;; shift the other object to bottom layer
            (not (at ?otherobj ?l middle)) ;; other object is no longer at the middle layer
          )
        ) 
      )
      ;; forall objects that at the top layer and at the same location, shift them to the middle layer
      (forall (?otherobj - normal-item)
        (when (and
          (at ?otherobj ?l top) ;; other object is at the top layer at the same location
          (not (= ?otherobj ?o)) ;; other object is not the same as the object being picked up
          ;; meanwhile also inside the same container
          (not (exists (?container - furniture)
            (and
              (at ?container ?l top) ;; container is at the same location and top layer
              (containable ?container top) ;; container is containable at the top layer
            )
          ))
          ;; meanwhile middle should not have not container stuff such as table 
          (not (exists (?f - furniture)
            (and
              (at ?f ?l middle) ;; furniture is at the same location and middle layer
              (not (containable ?f middle)) ;; furniture is not containable at the middle layer
            )
          ))
        )
          (and
            (at ?otherobj ?l middle) ;; shift the other object to middle layer
            (not (at ?otherobj ?l top)) ;; other object is no longer at the top layer
          )
        ) 
      )
      (increase (total-cost) 10)

    )
  )

  (:action pickup_1 ;; 1 means middle layer
    :parameters (?ag - agent ?o - normal-item ?l - location)
    :precondition (and
      (pickable ?o) ;; object is pickable
      (inreachofrobot ?ag ?o) ;; agent is in reach of the object
      (not (inhandofrobot ?ag ?o)) ;; agent is not already holding the object
      (at ?o ?l middle) ;; object is at the location and middle layer
      (not (exists (?c - furniture)
        (and
          (at ?c ?l middle) ;; container is at the same location and middle layer
          (containable ?c middle) ;; container is containable at the middle layer
        )
      ))
      ;; no other objects currently inhandofrobot
      (not (exists (?otherobj - normal-item)
        (and 
          (inhandofrobot ?ag ?otherobj) ;; agent is holding another object
        )
      ))
    )
    :effect (and
      (not (at ?o ?l middle)) ;; object is no longer at the location and middle layer
      (inhandofrobot ?ag ?o) ;; agent is now holding the object
      ;; forall objects that at the top layer and at the same location, shift them to middle layer
      (forall (?otherobj - normal-item)
        (when (and
          (at ?otherobj ?l top) ;; other object is at the top layer at the same location
          (not (= ?otherobj ?o)) ;; other object is not the same as the object being picked up
          ;; meanwhile also inside the same container
          (not (exists (?container - furniture)
            (and
              (at ?container ?l top) ;; container is at the same location and top layer
              (containable ?container top) ;; container is containable at the top layer
            )
          ))
        )
          (and
            (at ?otherobj ?l middle) ;; shift the other object to middle layer
            (not (at ?otherobj ?l top)) ;; other object is no longer at the top layer
          )
        ) 
      )
      (increase (total-cost) 10)

    )
  
  )

 

  (:action pickup_2
    :parameters (?ag - agent ?o - normal-item ?l - location)
    :precondition (and
      (pickable ?o) ;; object is pickable
      (inreachofrobot ?ag ?o) ;; agent is in reach of the object
      (not (inhandofrobot ?ag ?o)) ;; agent is not already holding the object
      (at ?o ?l top) ;; object is at the location and top layer
      ;; no other objects currently inhandofrobot
      (not (exists (?otherobj - normal-item)
        (and 
          (inhandofrobot ?ag ?otherobj) ;; agent is holding another object
        )
      ))
      (exists (?c - furniture)
        (imply
          (and
            (at ?c ?l top) ;; container is at the same location and top layer
            (containable ?c top) ;; container is containable at the top layer
            (openable ?c) ;; container is openable
          )
          (is-opened ?c) ;; if the container is openable, then it must be opened
        )
      )
    )
    :effect (and
      (not (at ?o ?l top)) ;; object is no longer at the location and top layer
      (inhandofrobot ?ag ?o) ;; agent is now holding the object
      (increase (total-cost) 10)

    )
  )

  ;; ── Slice  ──────────────────────────────────────────────────────────
  (:action slice
    :parameters (?ag - agent ?food - normal-item ?k - normal-item ?loc - location ?dim - dimension)
    :precondition (and
      (inhandofrobot ?ag ?k) ;; agent is holding a slicing tool
      (inreachofrobot ?ag ?food) ;; agent is in reach of the food
      (at ?food ?loc ?dim) ;; food is at the location and dimension
      (sliceable ?food) ;; food is sliceable
      (not (is-sliced ?food)) ;; food is not already sliced
      (slicer ?k) ;; the tool is a slicer
      (not (exists (?cnt - furniture)
        (and
          (containable ?cnt ?dim) ;; the container can contain at that dimension
          (at ?cnt ?loc ?dim) ;; the container is at the same location and dimension as the object
          (openable ?cnt) ;; the container is openable
          (not (is-opened ?cnt)) ;; the container is not opened
        )
      ))
    )
    :effect (and
      (is-sliced ?food) ;; food is now sliced
      (increase (total-cost) 10)
    )
  )



  ;; ── Toggle furniture (stove-burner, sink-tap, lamp…) ────────────────────────
  (:action toggle
    :parameters (?ag - agent ?f - item-and-furniture)
    :precondition (and
      (inreachofrobot ?ag ?f) ;; agent is in reach of the furniture
      (toggleable ?f) ;; furniture is toggleable
    )
    :effect (and
      (when (is-toggled ?f) (not (is-toggled ?f))) ;; if the furniture is toggled, then it is not toggled anymore
      (when (not (is-toggled ?f)) (is-toggled ?f)) ;; if the furniture is not toggled, then it is toggled now
      (increase (total-cost) 10)
    )
  )

  

  

  

  




)
