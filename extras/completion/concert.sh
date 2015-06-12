_concert()
{
    local cur prev opts base
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="edit import export init log rm mv cp show start docs"

    case "${prev}" in
        init|docs)
            local more=$(for x in `concert ${prev} --help | grep "  --" - | awk '{ print $1 }'`; do echo ${x}; done)
            COMPREPLY=($(compgen -W "${more}" -- ${cur}))
            return 0
            ;;
        edit|log|rm|start|mv|cp|export)
            local sessions=$(for x in `concert show | grep "  " - | awk '{ print $1 }'`; do echo ${x}; done)
            COMPREPLY=($(compgen -W "${sessions}" -- ${cur}))
            return 0
            ;;
        *)
        ;;
    esac

    COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
    return 0
}

complete -F _concert concert
